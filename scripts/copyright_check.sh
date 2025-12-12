#!/bin/bash

set -o pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

git fetch --unshallow &> /dev/null

currentYear=$(date +'%Y')

declare -a fileextensions=("py")
declare -a ignore=()
if [ -f .copyrightignore ]; then
  IFS=$'\n' read -d '' -r -a ignore < .copyrightignore
fi

echo "Ignored files: "
for e in "${ignore[@]}"; do echo "    - $e"; done

function shouldCheck() {
  local filename=$(basename "$1")
  local fileExtension="${filename##*.}"
  if [ ! -f "$1" ]; then
    return 1
  fi
  for e in "${ignore[@]}"; do [[ "$1" =~ $e && "$e" != "" ]] && return 1; done
  for e in "${fileextensions[@]}"; do [[ "$e" == "$fileExtension" ]] && return 0; done
  return 1
}

for filename in $(git ls-tree -r --name-only HEAD .); do
    shouldCheck "$filename"
    if [[ $? -eq 0 ]]; then
      copyrightFile="scripts/copyrights.py"

      copyrightYear=$(cat $filename | grep -m1 "Copyright IBM" | sed -En "s/.*Copyright IBM Corp\. ([0-9]+[-]){0,1}[-]?([0-9]+){0,1}.*$/\2/p")
      creationDate=$(git log --follow --format="%cd" --date=short -- $filename | tail -1)
      creationYear=${creationDate%%-*}
      copyrightYearCreate=$(cat $filename | grep -m1 "Copyright IBM" | sed -En "s/.*Copyright IBM Corp\. ([0-9]+){0,1}[-]?([0-9]+){0,1}.*$/\1/p")
      copyrightYearCreate=$(echo $copyrightYearCreate | sed -En "s/([0-9]+).*/\1/p")

      newCopyrightDates=$currentYear
      if [[ "$creationYear" != "$currentYear" ]]; then
        newCopyrightDates="$creationYear-$currentYear"
      fi

      if [[ "$copyrightYear" != "$currentYear" ]]; then
          if [[ $1 == "--fix" ]]; then
              if [ -z "${copyrightYear}" ] && [ -z "${copyrightYearCreate}" ]; then
                echo "Adding copyright to $filename"
                cat $copyrightFile $filename | sed -E "s/^(.*Copyright IBM Corp\. )COPYRIGHT_DATES(.*)$/\1""$newCopyrightDates""\2/" > ${filename}.bak
                mv ${filename}.bak ${filename}
              else
                echo "Updating $filename"
                sed -E -i.bak "s/^(.*Copyright IBM Corp\. [^0-9]*)[0-9]+[,-]?[ ]?[0-9]*([^0-9]*)$/\1""$newCopyrightDates""\2/" $filename
                rm ${filename}.bak
              fi
          else
            if [ -z "${copyrightYear}" ] && [ -z "${copyrightYearCreate}" ]; then
              echo "Copyright missing from $filename"
            else
              echo -e "${RED}Copyright needs to be updated for: ${filename}${NC}" >&2
            fi
            fail=true
          fi
      else
        if [[ "$creationYear" != "$copyrightYearCreate" ]]; then
            if [[ $1 == "--fix" ]]; then
              if [ -z "${copyrightYearCreate}" ] && [ -z "${copyrightYear}" ]; then
                echo "Adding copyright to $filename"
                cat $copyrightFile $filename | sed -E "s/^(.*Copyright IBM Corp\. )COPYRIGHT_DATES(.*)$/\1""$newCopyrightDates""\2/" > ${filename}.bak
                mv ${filename}.bak ${filename}
              else
                echo "Updating: $filename"
                sed -E -i.bak "s/^(.*Copyright IBM Corp\. [^0-9]*)[0-9]+[,-]?[0-9]*([^0-9]*)$/\1""$newCopyrightDates""\2/" $filename
                rm ${filename}.bak
              fi
            else
                echo -e "${RED}Copyright needs to be updated for: ${filename}${NC}" >&2
                if [[ -n "$creationDate" ]]; then
                  if [[ -n  "$copyrightYearCreate" ]]; then
                      echo "Created: ${creationDate} and written as ${copyrightYearCreate}."
                  else
                      echo "Created: ${creationDate} and missing creation year in file."
                  fi
                fi
                fail=true
            fi
        fi
      fi
    fi
done

if [[ "$fail" ]]; then
    echo -e "\n${RED}Correct copyrights with './copyright_check.sh --fix' parameter${NC}"
    exit 1
else
    echo -e "${GREEN}Copyright up to date :)${NC}"
fi
