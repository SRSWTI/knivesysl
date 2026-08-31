#!/usr/bin/env bash
# Pull every git repo under vendor/ and write a daily markdown report.
#
# Discovers repos by scanning vendor/ for `.git` entries -- both real repos
# (`.git/` directories) and nested submodule gitlinks (`.git` files, e.g.
# flashinfer/3rdparty/cutlass is a submodule of flashinfer's own history,
# not ours; vendor/ itself has no .gitmodules and is fully gitignored by
# the parent repo). Any depth, so vendor/intel-xpu/vllm-xpu/vllm-xpu-kernels
# is found the same as a top-level vendor/cutlass. Never descends into a
# directory literally named `.git` (that's git-internal storage, not
# another working tree to discover repos in).
#
# Per repo, in order:
#   1. Skip (report, don't touch) if `git rev-parse HEAD` itself fails --
#      not a valid checkout at all (seen in the wild: flashinfer copied its
#      own 3rdparty/{cccl,cutlass,spdlog} into flashinfer/data/ without
#      fixing the relative `gitdir:` pointer -- "not a git repository").
#   2. Skip (report, don't touch) if the working tree is dirty -- never
#      clobbers uncommitted local changes.
#   3. Skip (report, don't touch) if HEAD is detached -- nothing sane to
#      fast-forward onto.
#   4. `git pull --ff-only`, bounded by PULL_TIMEOUT_S. Never merges, never
#      rebases, never force-resets: on a genuine divergence this fails
#      loudly and the repo is reported failed, not silently mangled.
#   5. If HEAD moved, `git log <old>..<new>` for the commit list (hash,
#      date, author, subject) and `git diff-tree --name-status` per commit
#      for the changed-files list (capped), with a merge-commit note where
#      diff-tree legitimately shows nothing.
#
# Usage:
#   scripts/vendor_pull_report.sh                    # vendor/ under the repo root
#   scripts/vendor_pull_report.sh --vendor-dir PATH
#   scripts/vendor_pull_report.sh --out-dir PATH
#   scripts/vendor_pull_report.sh --dry-run           # discover + report only, no pull
#
# Cron (every morning at 07:00, output also captured to a log):
#   0 7 * * * cd /home/shooting-brake007/srswti/shooting-brake && \
#       scripts/vendor_pull_report.sh >> docs/vendor-reports/cron.log 2>&1
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor"
OUT_DIR="$REPO_ROOT/docs/vendor-reports"
DRY_RUN=0
PULL_TIMEOUT_S=180
MAX_COMMITS_LISTED=50
MAX_FILES_PER_COMMIT=30
FS=$'\x1f'   # field separator -- immune to commit-message quoting
RS=$'\x1e'   # record separator

while [ $# -gt 0 ]; do
    case "$1" in
        --vendor-dir) VENDOR_DIR="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ ! -d "$VENDOR_DIR" ]; then
    echo "error: $VENDOR_DIR is not a directory" >&2
    exit 1
fi

# -- discover repos -----------------------------------------------------------
# Prune descent into any directory named .git (git-internal storage);
# record both .git directories (real repos) and .git files (gitlinks).
mapfile -t GIT_ENTRIES < <(
    find "$VENDOR_DIR" \
        \( -type d -name .git -print -prune \) -o \
        \( -type f -name .git -print \) \
    | sort
)
REPOS=()
for g in "${GIT_ENTRIES[@]:-}"; do
    [ -n "$g" ] && REPOS+=("$(dirname "$g")")
done

echo "found ${#REPOS[@]} repos under $VENDOR_DIR"

mkdir -p "$OUT_DIR"
DATE_TAG="$(date +%Y-%m-%d)"
RUN_STAMP="$(date '+%Y-%m-%d %H:%M:%S %Z')"
OUT_FILE="$OUT_DIR/$DATE_TAG.md"
ROWS_TMP="$(mktemp)"
UPDATED_TMP="$(mktemp)"
FAILED_TMP="$(mktemp)"
SKIPPED_TMP="$(mktemp)"
BROKEN_TMP="$(mktemp)"
trap 'rm -f "$ROWS_TMP" "$UPDATED_TMP" "$FAILED_TMP" "$SKIPPED_TMP" "$BROKEN_TMP"' EXIT

N_UPDATED=0; N_NOCHANGE=0; N_SKIPPED=0; N_FAILED=0; N_BROKEN=0; N_COMMITS=0
T0=$(date +%s)

# Per-commit changed-files list, with a merge-commit fallback: diff-tree
# without -m/-c legitimately shows nothing for a merge (no single linear
# parent diff), which would otherwise look identical to "changed nothing".
commit_files() {
    local repo="$1" sha="$2" out parents
    out="$(git -C "$repo" diff-tree --no-commit-id --name-status -r "$sha" 2>/dev/null)"
    if [ -z "$out" ]; then
        parents="$(git -C "$repo" log -1 --format=%P "$sha" 2>/dev/null)"
        if [ "$(wc -w <<<"$parents")" -gt 1 ]; then
            echo "(merge commit -- no single-parent diff to show)"
            return
        fi
    fi
    printf '%s\n' "$out"
}

row() { # rel status branch commits before after
    printf '%s|%s|%s|%s|%s|%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >> "$ROWS_TMP"
}

for repo in "${REPOS[@]}"; do
    rel="${repo#"$REPO_ROOT"/}"
    printf '  %s ... ' "$rel"

    remote="$(git -C "$repo" remote get-url origin 2>/dev/null)"
    [ -z "$remote" ] && remote="(no origin)"

    before_raw="$(git -C "$repo" rev-parse HEAD 2>&1)"
    if [ $? -ne 0 ]; then
        detail="$(head -1 <<<"$before_raw")"
        printf 'broken (%s)\n' "$detail"
        echo "- \`$rel\`: $detail" >> "$BROKEN_TMP"
        row "$rel" broken "(detached)" - "?" "?"
        N_BROKEN=$((N_BROKEN + 1))
        continue
    fi
    before="${before_raw:0:12}"

    branch="$(git -C "$repo" symbolic-ref --short -q HEAD 2>/dev/null)"
    detached=0
    [ -z "$branch" ] && { branch="(detached)"; detached=1; }

    if [ -n "$(git -C "$repo" status --porcelain 2>/dev/null)" ]; then
        printf 'skipped-dirty (%s)\n' "$before"
        echo "- \`$rel\` (skipped-dirty): uncommitted local changes -- not touched" >> "$SKIPPED_TMP"
        row "$rel" skipped-dirty "$branch" - "$before" "$before"
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi
    if [ "$detached" -eq 1 ]; then
        printf 'skipped-detached (%s)\n' "$before"
        echo "- \`$rel\` (skipped-detached): detached HEAD -- not touched" >> "$SKIPPED_TMP"
        row "$rel" skipped-detached "$branch" - "$before" "$before"
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        printf 'dry-run (%s)\n' "$before"
        row "$rel" dry-run "$branch" - "$before" "$before"
        continue
    fi

    pull_err="$(timeout "$PULL_TIMEOUT_S" git -C "$repo" pull --ff-only 2>&1)"
    if [ $? -ne 0 ]; then
        detail="$(tail -1 <<<"$pull_err")"
        printf 'failed (%s)\n' "$before"
        echo "- \`$rel\`: $detail" >> "$FAILED_TMP"
        row "$rel" failed "$branch" - "$before" "$before"
        N_FAILED=$((N_FAILED + 1))
        continue
    fi

    after="$(git -C "$repo" rev-parse --short=12 HEAD)"

    if [ "$before" = "$after" ]; then
        printf 'no-change (%s)\n' "$before"
        row "$rel" no-change "$branch" - "$before" "$after"
        N_NOCHANGE=$((N_NOCHANGE + 1))
        continue
    fi

    printf 'updated (%s -> %s)\n' "$before" "$after"
    N_UPDATED=$((N_UPDATED + 1))

    {
        echo "### \`$rel\`"
        echo "remote: $remote · branch: $branch · \`$before\` -> \`$after\`"
        echo
        commit_count=0
        # git appends its own trailing newline after each --format entry
        # regardless of what's in the format string; none of the chosen
        # fields (hash/name/date/subject) are legitimately multi-line, so
        # stripping all newlines leaves RS as the sole record boundary.
        while IFS="$FS" read -r -d "$RS" full short author cdate subject; do
            [ -z "$full" ] && continue
            commit_count=$((commit_count + 1))
            [ "$commit_count" -gt "$MAX_COMMITS_LISTED" ] && break
            N_COMMITS=$((N_COMMITS + 1))
            echo "- **$short** $subject  "
            echo "  $author · $cdate"
            files="$(commit_files "$repo" "$full")"
            if [ -n "$files" ]; then
                echo '  ```'
                nfiles=0
                while IFS= read -r fline; do
                    nfiles=$((nfiles + 1))
                    [ "$nfiles" -le "$MAX_FILES_PER_COMMIT" ] && echo "  $fline"
                done <<<"$files"
                [ "$nfiles" -gt "$MAX_FILES_PER_COMMIT" ] &&
                    echo "  ... +$((nfiles - MAX_FILES_PER_COMMIT)) more files"
                echo '  ```'
            fi
        done < <(git -C "$repo" log "$before..$after" \
                  --format="%H${FS}%h${FS}%an${FS}%aI${FS}%s${RS}" | tr -d '\n')
        echo
    } >> "$UPDATED_TMP"

    row "$rel" updated "$branch" "$commit_count" "$before" "$after"
done

ELAPSED=$(( $(date +%s) - T0 ))

{
    echo "# Vendor pull report -- $DATE_TAG"
    echo
    echo "Run: $RUN_STAMP (took ${ELAPSED}s) · ${#REPOS[@]} repos · **$N_UPDATED updated** ($N_COMMITS commits) · $N_NOCHANGE unchanged · $N_SKIPPED skipped · $N_FAILED failed · $N_BROKEN broken"
    echo
    echo "| repo | status | branch | commits | before -> after |"
    echo "|---|---|---|---|---|"
    while IFS='|' read -r rel status branch commits before after; do
        echo "| \`$rel\` | $status | $branch | $commits | \`$before\` -> \`$after\` |"
    done < <(sort "$ROWS_TMP")
    echo

    if [ "$N_UPDATED" -gt 0 ]; then
        echo "## Updated"
        echo
        cat "$UPDATED_TMP"
    fi
    if [ -s "$FAILED_TMP" ]; then
        echo "## Failed"
        echo
        cat "$FAILED_TMP"
        echo
    fi
    if [ -s "$BROKEN_TMP" ]; then
        echo "## Broken (not a valid git checkout, left alone)"
        echo
        cat "$BROKEN_TMP"
        echo
    fi
    if [ -s "$SKIPPED_TMP" ]; then
        echo "## Skipped"
        echo
        cat "$SKIPPED_TMP"
        echo
    fi
} > "$OUT_FILE"

echo
echo "report -> $OUT_FILE"

[ "$N_FAILED" -gt 0 ] && exit 1
exit 0
