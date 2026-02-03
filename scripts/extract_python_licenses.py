#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extract Python package licenses for legal compliance.
Creates individual license files for each installed package.
"""
import os
import sys
from pathlib import Path
import re

def extract_licenses(venv_path, output_dir):
    """Extract licenses from Python packages."""
    site_packages = Path(venv_path) / "lib" / "python3.13" / "site-packages"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    manifest = []
    packages_with_license_files = 0
    packages_with_metadata = 0
    total_packages = 0
    
    # Find all dist-info directories
    dist_info_dirs = sorted(site_packages.glob("*.dist-info"))
    
    print(f"Found {len(dist_info_dirs)} Python packages")
    
    for dist_info in dist_info_dirs:
        total_packages += 1
        package_name = dist_info.name.replace(".dist-info", "")
        safe_name = re.sub(r'[^\w\-.]', '_', package_name)
        
        license_collected = False
        license_sources = []
        
        # 1. Copy existing LICENSE files
        for license_pattern in ['LICENSE*', 'COPYING*', 'NOTICE*']:
            for license_file in dist_info.glob(license_pattern):
                dest_file = output_path / f"{safe_name}.{license_file.name}"
                try:
                    with open(license_file, 'rb') as src:
                        content = src.read()
                    with open(dest_file, 'wb') as dst:
                        dst.write(content)
                    license_sources.append(f"FILE:{license_file.name}")
                    license_collected = True
                    packages_with_license_files += 1
                except Exception as e:
                    print(f"Warning: Failed to copy {license_file}: {e}", file=sys.stderr)
        
        # Also check licenses subdirectory (PEP 639 format)
        licenses_dir = dist_info / "licenses"
        if licenses_dir.exists():
            for license_file in licenses_dir.rglob("*"):
                if license_file.is_file():
                    dest_file = output_path / f"{safe_name}.licenses.{license_file.name}"
                    try:
                        with open(license_file, 'rb') as src:
                            content = src.read()
                        with open(dest_file, 'wb') as dst:
                            dst.write(content)
                        license_sources.append(f"LICENSES_DIR:{license_file.name}")
                        license_collected = True
                    except Exception as e:
                        print(f"Warning: Failed to copy {license_file}: {e}", file=sys.stderr)
        
        # 2. Extract from METADATA file
        metadata_file = dist_info / "METADATA"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8', errors='ignore') as f:
                    metadata_content = f.read()
                
                # Extract License field
                license_match = re.search(r'^License:\s*(.+)$', metadata_content, re.MULTILINE)
                if license_match:
                    license_text = license_match.group(1).strip()
                    
                    # Extract classifier licenses
                    classifiers = re.findall(r'^Classifier:.*?License\s*::\s*(.+)$', 
                                           metadata_content, re.MULTILINE)
                    
                    # Create license info file
                    license_info_file = output_path / f"{safe_name}.LICENSE_INFO.txt"
                    with open(license_info_file, 'w', encoding='utf-8') as f:
                        f.write(f"Package: {package_name}\n")
                        f.write(f"Source: dist-info/METADATA\n")
                        f.write(f"\nLicense: {license_text}\n")
                        
                        if classifiers:
                            f.write(f"\nLicense Classifiers:\n")
                            for classifier in classifiers:
                                f.write(f"  - {classifier.strip()}\n")
                        
                        # Include full metadata for reference
                        f.write(f"\n{'='*60}\n")
                        f.write(f"Full METADATA:\n")
                        f.write(f"{'='*60}\n")
                        f.write(metadata_content)
                    
                    license_sources.append("METADATA")
                    license_collected = True
                    packages_with_metadata += 1
                
            except Exception as e:
                print(f"Warning: Failed to read metadata for {package_name}: {e}", file=sys.stderr)
        
        # Record in manifest
        if license_collected:
            manifest.append(f"{package_name}\t{','.join(license_sources)}")
        else:
            manifest.append(f"{package_name}\tNO_LICENSE_FOUND")
            print(f"Warning: No license found for {package_name}", file=sys.stderr)
    
    # Write manifest
    manifest_file = output_path / "python-licenses-manifest.txt"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        f.write("# Python Package License Manifest\n")
        f.write(f"# Total packages: {total_packages}\n")
        f.write(f"# Packages with LICENSE files: {packages_with_license_files}\n")
        f.write(f"# Packages with METADATA: {packages_with_metadata}\n")
        f.write("#\n")
        f.write("# Format: PACKAGE_NAME<TAB>LICENSE_SOURCES\n")
        f.write("#\n")
        for line in sorted(manifest):
            f.write(line + "\n")
    
    # Write summary
    summary_file = output_path / "SUMMARY.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Python Package License Extraction Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Python packages: {total_packages}\n")
        f.write(f"Packages with dedicated LICENSE files: {packages_with_license_files}\n")
        f.write(f"Packages with license in METADATA: {packages_with_metadata}\n")
        f.write(f"Packages missing license info: {total_packages - len([m for m in manifest if 'NO_LICENSE' not in m])}\n")
        f.write(f"\nAll license information has been extracted to: {output_path}\n")
        f.write(f"\nLicense files are named: PACKAGE_NAME.LICENSE*\n")
        f.write(f"License info files are named: PACKAGE_NAME.LICENSE_INFO.txt\n")
    
    print(f"\nLicense extraction complete!")
    print(f"Total packages: {total_packages}")
    print(f"Packages with LICENSE files: {packages_with_license_files}")
    print(f"Packages with METADATA info: {packages_with_metadata}")
    print(f"Output directory: {output_path}")

if __name__ == "__main__":
    venv_path = os.environ.get("VIRTUAL_ENV", "/workspace/.venv")
    output_dir = "/legal/python-licenses"
    
    extract_licenses(venv_path, output_dir)
