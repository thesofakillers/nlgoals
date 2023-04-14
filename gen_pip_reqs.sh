#!/usr/bin/env bash
# Set default output file name
output_file="requirements.txt"

# help message
usage() {
  echo "Usage: $0 [--all] [--with <group1,group2,...>] [--output <file_name>]"
  echo "Export Python package requirements using poetry"
  echo ""
  echo "Options:"
  echo "  --all                   Export packages from all groups"
  echo "  --with <group1,group2>  Export packages only from specified groups (comma-separated)"
  echo "  --output <file_name>    Save output to specified file (default: requirements.txt)"
  exit 1
}

groups_specified=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -w | --with)
      groups+=("${2}")
      groups_specified=true
      shift 2
      ;;
    -a | --all)
      all_groups=true
      groups_specified=true
      shift
      ;;
    -o | --output)
      output_file="${2}"
      shift 2
      ;;
    -h | --help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# If --all was specified, get all the groups from the pyproject.toml file
if [ "$all_groups" = true ]; then
  groups=($(grep -E "^\[tool\.poetry\.group\.[^]]+\]$" pyproject.toml | cut -d "." -f 4))
fi

# Generate requirements.txt file
# whether to use the --with flag
if [ "$groups_specified" = true ]; then
  poetry export --without-hashes $(printf -- "--with %s " "${groups[@]}") -f requirements.txt -o "$output_file"
else
  poetry export --without-hashes -f requirements.txt -o "$output_file"
fi

# add local package information to the end of the file
echo "# local package" >> "$output_file"
echo "-e ." >> "$output_file"
