# Configuration settings.
export POSTGRESQL_MAX_CONNECTIONS=${POSTGRESQL_MAX_CONNECTIONS:-100}
export POSTGRESQL_MAX_PREPARED_TRANSACTIONS=${POSTGRESQL_MAX_PREPARED_TRANSACTIONS:-0}

# Perform auto-tuning based on the container cgroups limits (only when the
# limits are set).
# Users can still override this by setting the POSTGRESQL_SHARED_BUFFERS
# and POSTGRESQL_EFFECTIVE_CACHE_SIZE variables.
if [[ "${NO_MEMORY_LIMIT:-}" == "true" || -z "${MEMORY_LIMIT_IN_BYTES:-}" ]]; then
    export POSTGRESQL_SHARED_BUFFERS=${POSTGRESQL_SHARED_BUFFERS:-32MB}
    export POSTGRESQL_EFFECTIVE_CACHE_SIZE=${POSTGRESQL_EFFECTIVE_CACHE_SIZE:-128MB}
else
    # Use 1/4 of given memory for shared buffers
    shared_buffers_computed="$(($MEMORY_LIMIT_IN_BYTES/1024/1024/4))MB"
    # Setting effective_cache_size to 1/2 of total memory would be a normal conservative setting,
    effective_cache="$(($MEMORY_LIMIT_IN_BYTES/1024/1024/2))MB"
    export POSTGRESQL_SHARED_BUFFERS=${POSTGRESQL_SHARED_BUFFERS:-$shared_buffers_computed}
    export POSTGRESQL_EFFECTIVE_CACHE_SIZE=${POSTGRESQL_EFFECTIVE_CACHE_SIZE:-$effective_cache}
fi

export POSTGRESQL_LOG_DESTINATION=${POSTGRESQL_LOG_DESTINATION:-}

export POSTGRESQL_RECOVERY_FILE=$HOME/openshift-custom-recovery.conf
export POSTGRESQL_CONFIG_FILE=$HOME/openshift-custom-postgresql.conf

postinitdb_actions=

# match . files when moving userdata below
shopt -s dotglob
# extglob enables the !(userdata) glob pattern below.
shopt -s extglob

function usage() {
  if [ $# == 1 ]; then
    echo >&2 "error: $1"
  fi

  cat >&2 <<EOF
For general container run, you must either specify the following environment
variables:
  POSTGRESQL_USER  POSTGRESQL_PASSWORD  POSTGRESQL_DATABASE
Or the following environment variable:
  POSTGRESQL_ADMIN_PASSWORD
Or both.

To migrate data from different PostgreSQL container:
  POSTGRESQL_MIGRATION_REMOTE_HOST (hostname or IP address)
  POSTGRESQL_MIGRATION_ADMIN_PASSWORD (password of remote 'postgres' user)
And optionally:
  POSTGRESQL_MIGRATION_IGNORE_ERRORS=yes (default is 'no')

Optional settings:
  POSTGRESQL_MAX_CONNECTIONS (default: 100)
  POSTGRESQL_MAX_PREPARED_TRANSACTIONS (default: 0)
  POSTGRESQL_SHARED_BUFFERS (default: 32MB)
EOF
  exit 1
}

function check_env_vars() {
  if [[ -v POSTGRESQL_USER || -v POSTGRESQL_PASSWORD || -v POSTGRESQL_DATABASE ]]; then
    # one var means all three must be specified
    [[ -v POSTGRESQL_USER && -v POSTGRESQL_PASSWORD && -v POSTGRESQL_DATABASE ]] || usage

    [ ${#POSTGRESQL_USER}     -le 63 ] || usage "PostgreSQL username too long (maximum 63 characters)"
    [ ${#POSTGRESQL_DATABASE} -le 63 ] || usage "Database name too long (maximum 63 characters)"
    postinitdb_actions+=",simple_db"
  fi

  if [ -v POSTGRESQL_ADMIN_PASSWORD ]; then
    postinitdb_actions+=",admin_pass"
  fi

  if [ -v POSTGRESQL_MIGRATION_REMOTE_HOST -a \
       -v POSTGRESQL_MIGRATION_ADMIN_PASSWORD ]; then
    postinitdb_actions+=",migration"
  fi

  case "$postinitdb_actions" in
    ,simple_db,admin_pass) ;;
    ,migration|,simple_db|,admin_pass) ;;
    *) usage ;;
  esac
}

# Make sure env variables don't propagate to PostgreSQL process.
function unset_env_vars() {
  unset POSTGRESQL_{DATABASE,USER,PASSWORD,ADMIN_PASSWORD}
}

# postgresql_master_addr lookups the 'postgresql-master' DNS and get list of the available
# endpoints. Each endpoint is a PostgreSQL container with the 'master' PostgreSQL running.
function postgresql_master_addr() {
  local service_name=${POSTGRESQL_MASTER_SERVICE_NAME:-postgresql-master}
  local endpoints=$(dig ${service_name} A +search | grep ";${service_name}" | cut -d ';' -f 2 2>/dev/null)
  # FIXME: This is for debugging (docker run)
  if [ -v POSTGRESQL_MASTER_IP ]; then
    endpoints=${POSTGRESQL_MASTER_IP:-}
  fi
  if [ -z "$endpoints" ]; then
    >&2 echo "Failed to resolve PostgreSQL master IP address"
    exit 3
  fi
  echo -n "$(echo $endpoints | cut -d ' ' -f 1)"
}

# Converts the version in format x.y or x.y.z to a number.
version2number ()
{
    local old_IFS=$IFS
    local to_print= depth=${2-3} width=${3-2} sum=0 one_part
    IFS='.'
    set -- $1
    while test $depth -ge 1; do
        depth=$(( depth - 1 ))
        part=${1-0} ; shift || :
        printf "%0${width}d" "$part"
    done
    IFS=$old_IFS
}

# On non-intel arches, data_sync_retry = off does not work
# Upstream discussion: https://www.postgresql.org/message-id/CA+mCpegfOUph2U4ZADtQT16dfbkjjYNJL1bSTWErsazaFjQW9A@mail.gmail.com
# Upstream changes that caused this issue:
# https://github.com/postgres/postgres/commit/483520eca426fb1b428e8416d1d014ac5ad80ef4
# https://github.com/postgres/postgres/commit/9ccdd7f66e3324d2b6d3dec282cfa9ff084083f1
# RHBZ: https://bugzilla.redhat.com/show_bug.cgi?id=1779150
# Special handle of data_sync_retry should handle only in some cases.
# These cases are: non-intel architectures, and version higher or equal 12.0, 10.7, 9.6.12
# Return value 0 means the hack is needed.
function should_hack_data_sync_retry() {
  [ "$(uname -m)" == 'x86_64' ] && return 1
  local version_number=$(version2number "$(pg_ctl -V | sed -e 's/^pg_ctl (PostgreSQL) //')")
  # this matches all 12.x and versions of 10.x where we need the hack
  [ "$version_number" -ge 100700 ] && return 0
  # this matches all 10.x that were not matched above
  [ "$version_number" -ge 100000 ] && return 1
  # this matches all 9.x where need the hack
  [ "$version_number" -ge 090612 ] && return 0
  # all rest should be older 9.x releases
  return 1
}

# New config is generated every time a container is created. It only contains
# additional custom settings and is included from $PGDATA/postgresql.conf.
function generate_postgresql_config() {
  envsubst \
      < "${CONTAINER_SCRIPTS_PATH}/openshift-custom-postgresql.conf.template" \
      > "${POSTGRESQL_CONFIG_FILE}"

  if [ "${ENABLE_REPLICATION}" == "true" ]; then
    envsubst \
        < "${CONTAINER_SCRIPTS_PATH}/openshift-custom-postgresql-replication.conf.template" \
        >> "${POSTGRESQL_CONFIG_FILE}"
  fi

  if should_hack_data_sync_retry ; then
    echo "data_sync_retry = on" >>"${POSTGRESQL_CONFIG_FILE}"
  fi

  # For easier debugging, allow users to log to stderr (will be visible
  # in the pod logs) using a single variable
  # https://github.com/sclorg/postgresql-container/issues/353
  if [ -n "${POSTGRESQL_LOG_DESTINATION:-}" ] ; then
    echo "log_destination = 'stderr'" >>"${POSTGRESQL_CONFIG_FILE}"
    echo "logging_collector = on" >>"${POSTGRESQL_CONFIG_FILE}"
    echo "log_directory = '$(dirname "${POSTGRESQL_LOG_DESTINATION}")'" >>"${POSTGRESQL_CONFIG_FILE}"
    echo "log_filename = '$(basename "${POSTGRESQL_LOG_DESTINATION}")'" >>"${POSTGRESQL_CONFIG_FILE}"
  fi
}

function generate_postgresql_recovery_config() {
  envsubst \
      < "${CONTAINER_SCRIPTS_PATH}/openshift-custom-recovery.conf.template" \
      > "${POSTGRESQL_RECOVERY_FILE}"
}

function initialize_database() {
  # Initialize the database cluster with utf8 support using environment variable 'LANG'.
  # https://www.postgresql.org/docs/16/locale.html
  initdb

  # PostgreSQL configuration.
  cat >> "$PGDATA/postgresql.conf" <<EOF

# Custom OpenShift configuration:
include '${POSTGRESQL_CONFIG_FILE}'
EOF

  # Access control configuration.
  # FIXME: would be nice-to-have if we could allow connections only from
  #        specific hosts / subnet
  cat >> "$PGDATA/pg_hba.conf" <<EOF

#
# Custom OpenShift configuration starting at this point.
#

# Allow connections from all hosts.
host all all all md5

# Allow replication connections from all hosts.
host replication all all md5
EOF
}

function create_users() {
  if [[ ",$postinitdb_actions," = *,simple_db,* ]]; then
    createuser "$POSTGRESQL_USER"
    createdb --owner="$POSTGRESQL_USER" "$POSTGRESQL_DATABASE"
  fi

  if [ -v POSTGRESQL_MASTER_USER ]; then
    createuser "$POSTGRESQL_MASTER_USER"
    echo "ALTER DATABASE postgres OWNER TO $POSTGRESQL_MASTER_USER;" | psql
    echo "GRANT ALL PRIVILEGES on DATABASE postgres TO $POSTGRESQL_MASTER_USER;" | psql
  fi
}

function set_pgdata ()
{
  export PGDATA=$HOME/data/userdata
  # create a subdirectory that the user owns
  mkdir -p "$PGDATA"
  # backwards compatibility case, we used to put the data here,
  # move it into our new expected location (userdata)
  if [ -e ${HOME}/data/PG_VERSION ]; then
    pushd "${HOME}/data"
    # move everything except the userdata directory itself, into the userdata directory.
    mv !(userdata) "userdata"
    popd
  fi
  # ensure sane perms for postgresql startup
  chmod 700 "$PGDATA"
}

function wait_for_postgresql_master() {
  while true; do
    master_fqdn=$(postgresql_master_addr)
    echo "Waiting for PostgreSQL master (${master_fqdn}) to accept connections ..."
    if [ -v POSTGRESQL_ADMIN_PASSWORD ]; then
      PGPASSWORD=${POSTGRESQL_ADMIN_PASSWORD} psql "postgresql://postgres@${master_fqdn}" -c "SELECT 1;" && return 0
    else
      PGPASSWORD=${POSTGRESQL_PASSWORD} psql "postgresql://${POSTGRESQL_USER}@${master_fqdn}/${POSTGRESQL_DATABASE}" -c "SELECT 1;" && return 0
    fi
    sleep 1
  done
}
