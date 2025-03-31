import re
import sys
from pathlib import Path

def bump_version(part="patch"):
    setup_path = Path("setup.py")
    setup_text = setup_path.read_text(encoding="utf-8")

    # 匹配 version="0.1.3" 这样的写法
    version_pattern = re.compile(r'version\s*=\s*["\'](\d+)\.(\d+)\.(\d+)["\']')
    match = version_pattern.search(setup_text)

    if not match:
        print("❌ Could not find a version string in setup.py.")
        sys.exit(1)

    major, minor, patch = map(int, match.groups())

    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        print("❌ Invalid version part. Use 'major', 'minor', or 'patch'.")
        sys.exit(1)

    new_version = f'{major}.{minor}.{patch}'
    print(f"🔁 Updating version: {match.group(0)} → version=\"{new_version}\"")

    setup_text = version_pattern.sub(f'version="{new_version}"', setup_text)
    setup_path.write_text(setup_text, encoding="utf-8")

    print("✅ setup.py version updated.")

if __name__ == "__main__":
    # 默认是 patch 更新，可以传入 'minor' 或 'major'
    bump_type = sys.argv[1] if len(sys.argv) > 1 else "patch"
    bump_version(bump_type)
