param(
    [Parameter(Mandatory=$true)][string]$Baseline,
    [Parameter(Mandatory=$true)][string]$Candidate
)
$ErrorActionPreference = 'Stop'
$Baseline = (Resolve-Path $Baseline).Path
$Candidate = (Resolve-Path $Candidate).Path

$cases = @(
    @{ name = 'default'; args = @('0','640','360','400','-1.4002','0','0.1','-0.4','0.6','OUT') },
    @{ name = 'julia';   args = @('1','640','360','300','0','0','1.5','-0.4','0.6','OUT') },
    @{ name = 'ss';      args = @('0','320','180','500','-0.743','0.131','0.01','0','0','OUT','--ss','2') },
    @{ name = 'palette'; args = @('0','400','300','200','-0.5','0','1.5','0','0','OUT','0.000000','000000','0.400000','FF0000','1.000000','FFFFFF') },
    @{ name = 'odd';     args = @('1','1001','333','250','0','0','1.2','0.285','0.01','OUT','--ss','3') }
)

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("cmpbase_" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
$failed = $false
try {
    foreach ($c in $cases) {
        $hashes = @()
        foreach ($who in @(@('b',$Baseline), @('c',$Candidate))) {
            $out = Join-Path $tmp ("$($c.name)_$($who[0]).png")
            $argv = $c.args | ForEach-Object { if ($_ -eq 'OUT') { $out } else { $_ } }
            & $who[1] @argv | Out-Null
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) { $hashes += 'ERR'; continue }
            $hashes += (Get-FileHash -Algorithm SHA256 $out).Hash
        }
        if ($hashes[0] -ne 'ERR' -and $hashes[0] -eq $hashes[1]) { Write-Host "PASS $($c.name)" }
        else { Write-Host "FAIL $($c.name)"; $failed = $true }
    }
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
if ($failed) { exit 1 } else { exit 0 }
