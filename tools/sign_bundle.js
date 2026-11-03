const { spawnSync } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs');

const keyPath = path.join(os.homedir(), '.tauri', 'clarity.key');
if (!fs.existsSync(keyPath)) {
  console.error('Private key not found at:', keyPath);
  process.exit(1);
}

const keyContent = fs.readFileSync(keyPath, 'utf8').trim();
let targetFile = process.argv[2];
if (!targetFile) {
  const tauriConf = JSON.parse(fs.readFileSync(path.join(__dirname, '../src-tauri/tauri.conf.json'), 'utf8'));
  targetFile = path.join(__dirname, `../src-tauri/target/release/bundle/nsis/Clarity_${tauriConf.version}_x64-setup.exe`);
}

console.log('Signing file:', targetFile);

const env = {
  ...process.env,
  TAURI_SIGNING_PRIVATE_KEY: keyContent,
  TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ''
};

const res = spawnSync('npx.cmd', ['tauri', 'signer', 'sign', `"${targetFile}"`], {
  encoding: 'utf8',
  env: env,
  shell: true,
  timeout: 60000
});

console.log(res.stdout);
if (res.stderr) console.error(res.stderr);

if (res.status !== 0) {
  process.exit(res.status || 1);
}
