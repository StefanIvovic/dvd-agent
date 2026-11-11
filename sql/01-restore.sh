#!/bin/bash
# Restores the dvdrental sample database on the container's first start (empty data volume).
set -euo pipefail
echo "restoring dvdrental sample database"
pg_restore --no-owner --role="$POSTGRES_USER" -U "$POSTGRES_USER" -d "$POSTGRES_DB" /seed/dvdrental.tar
