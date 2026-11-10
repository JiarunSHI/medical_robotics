param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('prepare','pretrain','pairs','finetune','evaluate','compare','test')]
    [string]$Step,
    [string]$Python = 'D:\anaconda\envs\pytorch\python.exe'
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    switch ($Step) {
        'prepare' { & $Python prepare.py --all-image-sizes }
        'pretrain' { & $Python train.py --stage pretrain --run tg3k_v1 }
        'pairs' { & $Python prepare_pairs.py }
        'finetune' { & $Python train.py --stage finetune --init runs/tg3k_v1/best.pt --manifest data/original_pairs_v3/local_pairs.csv --run local_original_v3 }
        'evaluate' { & $Python evaluate.py --checkpoint runs/local_original_v3/best.pt --manifest data/original_pairs_v3/final_test.csv --run test_original_v3 }
        'test' { & $Python -m unittest discover -s tests -v }
        'compare' { & $Python compare_models.py }
    }
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Step (exit $LASTEXITCODE)" }
} finally { Pop-Location }
