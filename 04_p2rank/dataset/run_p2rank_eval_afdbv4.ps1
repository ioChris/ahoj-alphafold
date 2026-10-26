# P2Rank eval-predict on the AFDB v4 re-alignment dataset (st100_run3_afdbv4), the two runs of July 2026:
#   1. SAS points as atoms (the native reference; July: test_output\UBS2_run3_dev_clean)
#   2. -site_eval_sas_pts_as_atoms 0 (classic residue-atom DCA; July: test_output\UBS2_run3_sasoff)
# Same build as July: p2rank_dev, version 2.6-alpha, bin\p2rank.jar of 2026-07-04 20:13 (md5 fb9961c5eb93453b1cdc36c23bd78800),
# config alphafold, Java 17 from the p2rankjre conda env. The "git head" line of P2Rank's params.txt is the head of
# the repository the command is run from, not P2Rank's (July: 1de6ffb = ahoj-alphafold), so it does not name the build.
# Only the dataset (.ds written by build_p2rank_ds.py) and -out_subdir differ.
# Output lands under <p2rank_dev>\test_output\<out_subdir>; the July folders are not touched.
param(
  [string]$PrankDir = $env:P2RANK_HOME,                    # the p2rank_dev build (holds prank.bat)
  [string]$JavaHome = $env:P2RANK_JAVA_HOME,               # Java 17 home, e.g. <p2rankjre conda env>\Library
  [Parameter(Mandatory = $true)][string]$Dataset,          # ...\st_100_intersection_geom\st_100_p2rank_intersection_geom.ds
  [string]$OutSubdir = "UBS2_run3_afdbv4"
)
$ErrorActionPreference = "Continue"
if (-not $PrankDir -or -not $JavaHome) {
  Write-Error "Pass -PrankDir and -JavaHome, or set P2RANK_HOME and P2RANK_JAVA_HOME."; exit 2
}
$env:JAVA_HOME = $JavaHome
$env:PATH = "$JavaHome\bin;$env:PATH"
$prank = "$PrankDir\prank.bat"
function Stamp($m) { "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m }

foreach ($d in @("$PrankDir\test_output\$OutSubdir", "$PrankDir\test_output\${OutSubdir}_sasoff")) {
  if (Test-Path $d) { Stamp "REFUSING: $d exists"; exit 2 }
}

Stamp "eval-predict CENTER_explicit, SAS points as atoms (native reference)"
& $prank eval-predict "$Dataset" -c alphafold -log_to_console 0 -out_subdir $OutSubdir -l CENTER_explicit -site_eval_center_method explicit -site_eval_sas_pts_as_atoms 1 -visualizations 0
Stamp "exit=$LASTEXITCODE"

Stamp "eval-predict CENTER_explicit, SAS points off (classic residue-atom DCA)"
& $prank eval-predict "$Dataset" -c alphafold -l CENTER_explicit -site_eval_center_method explicit -site_eval_sas_pts_as_atoms 0 -visualizations 0 -log_to_console 0 -threads 28 -out_subdir "${OutSubdir}_sasoff"
Stamp "exit=$LASTEXITCODE"
Stamp "done: $PrankDir\test_output\$OutSubdir and ${OutSubdir}_sasoff"
