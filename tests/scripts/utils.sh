# Wait for the ols api server to respond 200 on the readiness endpoint
# $1 - url of the ols server to poll
function wait_for_ols() {
    # Don't exit on error while polling the OLS server
    # Curl will return error exit codes until OLS is available
    set +e
    STARTED=0
    for i in {1..20}; do
        echo Checking OLS readiness, attempt "$i" of 20
        curl -sk --fail "$1/readiness"
        if [ $? -eq 0 ]; then
            STARTED=1
            break
        fi  
        sleep 6
    done
    set -e

    if [ $STARTED -ne 1 ]; then
        echo "Timed out waiting for OLS to start"
        exit 1
    fi
}
