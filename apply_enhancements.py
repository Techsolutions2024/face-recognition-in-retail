#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-apply enhancements to facere_gui.py
Tự động apply tất cả modifications
"""

import sys
import shutil
from pathlib import Path

print("=" * 70)
print("AUTO-APPLY ENHANCEMENTS TO FACERE_GUI.PY")
print("=" * 70)

# Check if original file exists
original_file = Path("facere_gui.py")
if not original_file.exists():
    print("❌ Error: facere_gui.py not found!")
    print("   Make sure you're in the correct directory.")
    sys.exit(1)

# Backup original file
backup_file = Path("facere_gui.py.backup")
if not backup_file.exists():
    print(f"\n📦 Creating backup: {backup_file}")
    shutil.copy2(original_file, backup_file)
    print("   ✅ Backup created successfully!")
else:
    print(f"\n⚠️  Backup already exists: {backup_file}")
    response = input("   Continue anyway? (y/n): ")
    if response.lower() != 'y':
        print("   ❌ Aborted.")
        sys.exit(0)

print("\n📝 Reading original file...")
with open(original_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"   ✅ Read {len(lines)} lines")

# Modifications
modifications = []

# 1. Add imports after line 45
print("\n1️⃣  Adding new imports...")
import_code = """
# NEW IMPORTS FOR EVENTS & CROPS MANAGEMENT
from datetime import datetime
try:
    from database import Database
    from events_manager import EventsManager
    from crops_manager import CropsManager
    from models import CustomerSegment
except ImportError as e:
    log.warning(f"Could not import new modules: {e}")
    log.warning("Events and Crops features will be disabled")

"""
modifications.append(('insert_after', 45, import_code))

# 2. Add new signals to VideoThread after line 54
print("2️⃣  Adding new signals to VideoThread...")
signals_code = """    # NEW: Signals for events and crops
    face_recognized_signal = pyqtSignal(dict)  # {'name': ..., 'id': ..., 'bbox': ..., 'confidence': ..., 'crop': ..., 'face_id': ...}
    unknown_face_signal = pyqtSignal(dict)  # {'bbox': ..., 'confidence': ..., 'crop': ...}

"""
modifications.append(('insert_after', 54, signals_code))

# 3. Replace face info appending in VideoThread.run() around line 159-164
print("3️⃣  Modifying VideoThread.run() to emit new signals...")
replace_code = """                            # Save face info
                            face_info = {
                                'crop': face_crop,
                                'label': label,
                                'confidence': confidence,
                                'bbox': (xmin, ymin, xmax, ymax)
                            }
                            faces_info.append(face_info)

                            # NEW: Emit events for database tracking
                            if identity.id != FaceIdentifier.UNKNOWN_ID:
                                # Get face_id from database
                                face_id = self.face_identifier.get_identity_label(identity.id)

                                # Emit recognized signal
                                self.face_recognized_signal.emit({
                                    'name': label,
                                    'id': identity.id,
                                    'face_id': face_id,
                                    'bbox': (xmin, ymin, xmax, ymax),
                                    'confidence': confidence,
                                    'crop': face_crop
                                })
                            else:
                                # Emit unknown signal
                                self.unknown_face_signal.emit({
                                    'bbox': (xmin, ymin, xmax, ymax),
                                    'confidence': 0.0,
                                    'crop': face_crop
                                })
"""
modifications.append(('replace', (159, 164), replace_code))

print("\n" + "=" * 70)
print("MODIFICATIONS TO APPLY:")
print("=" * 70)
for i, mod in enumerate(modifications, 1):
    if mod[0] == 'insert_after':
        print(f"{i}. Insert after line {mod[1]}: {mod[2].count(chr(10))} lines")
    elif mod[0] == 'replace':
        print(f"{i}. Replace lines {mod[1][0]}-{mod[1][1]}: {mod[2].count(chr(10))} lines")

print("\n⚠️  WARNING: This script applies PARTIAL modifications only!")
print("   For COMPLETE modifications, follow IMPLEMENTATION_GUIDE.md manually.")
print("   This script adds:")
print("   - New imports")
print("   - New VideoThread signals")
print("   - Modified event emission in VideoThread.run()")
print("\n   You still need to add:")
print("   - Database initialization in FaceRecognitionGUI")
print("   - Events Panel UI")
print("   - Event handler methods")
print("   - Camera Management dialog")

response = input("\n❓ Continue with partial application? (y/n): ")
if response.lower() != 'y':
    print("   ❌ Aborted.")
    sys.exit(0)

# Apply modifications
print("\n🔧 Applying modifications...")

# Create new content
new_lines = lines.copy()

# Apply in reverse order to maintain line numbers
for mod in reversed(modifications):
    if mod[0] == 'insert_after':
        line_num = mod[1]
        code = mod[2]
        new_lines.insert(line_num, code)
        print(f"   ✅ Inserted after line {line_num}")

    elif mod[0] == 'replace':
        start_line, end_line = mod[1]
        code = mod[2]
        # Remove old lines
        for _ in range(start_line, end_line + 1):
            if start_line < len(new_lines):
                new_lines.pop(start_line)
        # Insert new code
        new_lines.insert(start_line, code)
        print(f"   ✅ Replaced lines {start_line}-{end_line}")

# Write modified file
output_file = Path("facere_gui_partial.py")
print(f"\n💾 Writing to: {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"   ✅ Written {len(new_lines)} lines")

print("\n" + "=" * 70)
print("✅ PARTIAL MODIFICATIONS APPLIED SUCCESSFULLY!")
print("=" * 70)
print(f"\n📁 Files created:")
print(f"   - {backup_file} (backup of original)")
print(f"   - {output_file} (partially modified)")
print(f"\n⚠️  NEXT STEPS:")
print(f"   1. Review {output_file}")
print(f"   2. Follow IMPLEMENTATION_GUIDE.md for COMPLETE modifications:")
print(f"      - Add database attributes to FaceRecognitionGUI.__init__")
print(f"      - Add init_database() method")
print(f"      - Modify init_ui() to add Events Panel")
print(f"      - Add create_events_panel() method")
print(f"      - Add event handler methods")
print(f"      - Modify start_video() to connect signals")
print(f"      - Add Camera Management dialog")
print(f"\n   OR: Let me create fully modified facere_gui_enhanced.py for you!")
print("=" * 70)
