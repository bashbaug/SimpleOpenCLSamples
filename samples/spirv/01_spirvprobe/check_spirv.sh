#!/usr/bin/env bash
set -u

dir="spirv_dumps"
tool="./spirvkernelfromfile"
options="${options:-}"

if [[ ! -d "$dir" ]]; then
    echo "Error: directory not found: $dir"
    exit 1
fi

if [[ ! -x "$tool" ]]; then
    echo "Error: tool not found or not executable: $tool"
    exit 1
fi

shopt -s nullglob
files=("$dir"/*.spv)

if (( ${#files[@]} == 0 )); then
    echo "No .spv files found in $dir"
    exit 0
fi

for file in "${files[@]}"; do
    name="$(basename "$file")"

    if [[ -z "$options" ]]; then
        printf 'Running: %q --file=%q\n' "$tool" "$file"
        "$tool" --file="$file" > /dev/null 2>&1
    else
        printf 'Running: %q %q --file=%q\n' "$tool" "$options" "$file"
        "$tool" "$options" --file="$file" > /dev/null 2>&1
    fi
    rc=$?

    if (( rc == 0 )); then
        echo "SUCCESS: $name"
    else
        echo "ERROR (exit code $rc): $name"
    fi
done
