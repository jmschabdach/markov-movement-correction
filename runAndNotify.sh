#!/bin/bash

"$@" |& tee command_output.txt

LOC="DBMI"

MESSAGE=$'\nYour code has finished running on'
MESSAGE="$MESSAGE $LOC. You should go check on the results."

(echo "$@ $MESSAGE"; cat command_output.txt) > mail_notification.txt

mail -s "Code Completion Notification" jmschabdach@gmail.com < mail_notification.txt

rm mail_notification.txt
rm command_output.txt
