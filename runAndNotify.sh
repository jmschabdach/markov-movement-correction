#!/bin/bash

"$@"

LOC="CHP"

MESSAGE=$'\nYour code has finished running on '
MESSAGE="$MESSAGE $LOC. You should go check on the results."

echo "$@ $MESSAGE" | mail -s "Code Completion Notification" jmschabdach@gmail.com
