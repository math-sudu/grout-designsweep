<#
.SYNOPSIS
    Run a Python script inside the image built from the project Dockerfile.

.DESCRIPTION
    Mounts the project root (resolved from this script's location) as /work
    in the container, and runs python3 with the supplied arguments.

    The image already sets PYTHONPATH so dolfinx imports work.  The wrapper
    prepends /work/src inside the container while preserving the image path.

    Examples:
      .\run_in_dolfinx.ps1 scripts/prepare_support_response.py
      .\run_in_dolfinx.ps1 scripts/run_radial_support.py --workers 8
      .\run_in_dolfinx.ps1 scripts/check_support_profiles.py --critical-refinement
#>
[CmdletBinding(PositionalBinding=$false)]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Args
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$Image = 'grout-support-fe'

docker run --rm `
    -v "${ProjectRoot}:/work" `
    -w /work `
    -e OMP_NUM_THREADS=1 `
    -e OPENBLAS_NUM_THREADS=1 `
    $Image `
    bash -lc 'export PYTHONPATH="/work/src:${PYTHONPATH:-}"; python3 "$@"' bash @Args
exit $LASTEXITCODE
