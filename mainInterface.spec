# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['mainInterface.py'],
             pathex=['/home/gabriel/PycharmProjects/antropometria36_ui'],
             binaries=[],
             datas=[
			            ('distances/distortion_matrix.pkl', 'distances'),
                        ('distances/FacePointsExtractor.iml', 'distances'),
                        ("distances/haarcascade_eye.xml", "distances"),
                        ("distances/haarcascade_frontalface_alt2.xml", "distances"),
                        ("distances/mmod_human_face_detector.dat", "distances"),
                        ("distances/shape_predictor_68_face_landmarks.dat", "distances"),
                        ("models/rf.joblib", "models"), ("models/rf.sav", "models"),
                        ("images/usp.png", "images"), ("images/each.png", "images"),
                        ("output/README.md", "output")
	        ],
             hiddenimports=['PIL', 'PIL._imagingtk', 'PIL._tkinter_finder', 'sklearn.pipeline', 'sklearn.ensemble'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='Antropometria',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='Antropometria')
