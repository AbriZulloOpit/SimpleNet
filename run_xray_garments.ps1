if (-not $env:CONDA_PREFIX) {
    throw "Attiva prima l'ambiente Conda dirty-hands con: conda activate dirty-hands"
}

$Python = Join-Path $env:CONDA_PREFIX "python.exe"
if (-not (Test-Path $Python)) {
    throw "Python non trovato nell'ambiente Conda attivo: $Python"
}

Set-Location $PSScriptRoot
$DataPath = "..\xray-garments-simplenet"

& $Python main.py `
    --gpu 0 `
    --seed 0 `
    --log_group simplenet_xray_garments `
    --log_project XRayGarments_Results `
    --results_path results `
    --run_name xray_garments_run `
    net `
    -b wideresnet50 `
    -le layer2 `
    -le layer3 `
    --pretrain_embed_dimension 1536 `
    --target_embed_dimension 1536 `
    --patchsize 3 `
    --meta_epochs 15 `
    --embedding_size 256 `
    --gan_epochs 4 `
    --noise_std 0.015 `
    --dsc_hidden 1024 `
    --dsc_layers 2 `
    --dsc_margin .5 `
    --pre_proj 1 `
    dataset `
    --batch_size 8 `
    --resize 329 `
    --imagesize 288 `
    -d xray-garments `
    mvtec `
    $DataPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}