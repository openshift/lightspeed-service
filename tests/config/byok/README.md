# Generate the SkyNet BYOK images

Run from the `lightspeed-service` repository root after logging in with
`podman login registry.redhat.io`.

Prepare two copies of [skynet.md](skynet.md) with identical content and title,
but citation URLs differing in version:

```bash
TOOL_IMAGE=registry.redhat.io/openshift-lightspeed-tech-preview/lightspeed-rag-tool-rhel9:latest
AUTHFILE=${REGISTRY_AUTH_FILE:-$XDG_RUNTIME_DIR/containers/auth.json}
WORKDIR=$(mktemp -d)
mkdir -p "$WORKDIR/1.1" "$WORKDIR/1.2" "$WORKDIR/output"

for version in 1.1 1.2; do
  {
    printf -- '---\ntitle: "SkyNet Operations Guide"\nurl: "https://docs.cyberdyne.example.invalid/skynet/%s/operations"\n---\n\n' "$version"
    cat tests/config/byok/skynet.md
  } > "$WORKDIR/$version/skynet.md"
done
```

Run the existing BYOK RAG tool once for each copy:

```bash
podman run --rm --device=/dev/fuse \
  -e BYOK_TOOL_IMAGE="$TOOL_IMAGE" \
  -e OUT_IMAGE_TAG=byok-skynet-1.1 \
  -v "$AUTHFILE:/run/user/0/containers/auth.json:ro,z" \
  -v "$WORKDIR/1.1:/markdown:Z" \
  -v "$WORKDIR/output:/output:Z" \
  "$TOOL_IMAGE"

podman run --rm --device=/dev/fuse \
  -e BYOK_TOOL_IMAGE="$TOOL_IMAGE" \
  -e OUT_IMAGE_TAG=byok-skynet-1.2 \
  -v "$AUTHFILE:/run/user/0/containers/auth.json:ro,z" \
  -v "$WORKDIR/1.2:/markdown:Z" \
  -v "$WORKDIR/output:/output:Z" \
  "$TOOL_IMAGE"
```

The tool writes `byok-skynet-1.1.tar` and `byok-skynet-1.2.tar` to
`$WORKDIR/output`. No custom build script or Containerfile is needed.
