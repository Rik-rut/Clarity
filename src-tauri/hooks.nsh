; Clarity Desktop NSIS Installer Hooks
;
; The installer owns first-run provisioning, and everything it creates lands in
; the folder the user just chose. The app computes the same location from its own
; executable path (AppState::default_app_data_dir), so the two cannot disagree —
; which is exactly what happened when the hook used $LOCALAPPDATA\Clarity while
; the app looked in <drive>\Clarity-data and downloaded the models a second time.

Var DataDir
Var TensorrtChoice

!macro NSIS_HOOK_PREINSTALL
  DetailPrint "Preparing Clarity installation..."
!macroend
!macro NSIS_HOOK_POSTINSTALL
  ; ---- where everything lives ----------------------------------------------
  ; $INSTDIR is the folder the user chose on the directory page. It is writable
  ; in the normal case; a Program Files install is not, and must fall back
  ; rather than fail during a multi-gigabyte download.
  StrCpy $DataDir "$INSTDIR"
  FileOpen $0 "$INSTDIR\.clarity-write-test" w
  IfErrors 0 clarity_writable
    StrCpy $DataDir "$LOCALAPPDATA\Clarity"
    CreateDirectory "$DataDir"
    DetailPrint "[WARN] $INSTDIR is not writable; provisioning into $DataDir"
  clarity_writable:
  FileClose $0
  Delete "$INSTDIR\.clarity-write-test"

  IfFileExists "$INSTDIR\resources\uv.exe" clarity_have_uv 0
    DetailPrint "[WARN] resources\uv.exe is missing; cannot provision."
    Goto clarity_hook_done
  clarity_have_uv:

  ; ---- 1/4 Python runtime ---------------------------------------------------
  DetailPrint "[1/4] Installing the Python 3.11 runtime into $DataDir\python ..."
  System::Call 'kernel32::SetEnvironmentVariable(t "UV_PYTHON_INSTALL_DIR", t "$DataDir\python")'
  nsExec::ExecToLog '"$INSTDIR\resources\uv.exe" python install 3.11 --no-bin --install-dir "$DataDir\python"'
  Pop $0
  IntCmp $0 0 clarity_python_ok clarity_failed clarity_failed
  clarity_python_ok:

  ; ---- 2/4 TensorRT ---------------------------------------------------------
  ; The app picks TensorRT by itself whenever `import tensorrt` works, so this
  ; is only about whether the package gets installed at all.
  StrCpy $TensorrtChoice "no"
  nsExec::ExecToStack 'nvidia-smi -L'
  Pop $0
  Pop $1
  IntCmp $0 0 clarity_nvidia clarity_no_nvidia clarity_no_nvidia
  clarity_nvidia:
    DetailPrint "  NVIDIA GPU detected: $1"
    IfSilent clarity_trt_silent 0
      MessageBox MB_YESNO|MB_ICONQUESTION "Install TensorRT acceleration for your NVIDIA GPU?$\n$\nInterpolation runs considerably faster. Adds roughly 1-2 GB to this download." IDNO clarity_trt_no
      StrCpy $TensorrtChoice "yes"
      Goto clarity_no_nvidia
    clarity_trt_silent:
      ; A silent install is an unattended deployment: give it the acceleration.
      StrCpy $TensorrtChoice "yes"
    clarity_trt_no:
  clarity_no_nvidia:

  ; ---- 3/4 engine, dependencies, models -------------------------------------
  ; One implementation, shared with the app's repair path. `uv run --no-project`
  ; executes the script with the managed interpreter without creating an
  ; environment of its own; provision.py finds the package via its own path.
  DetailPrint "[2/4] Creating the environment and installing PyTorch (TensorRT: $TensorrtChoice) ..."
  DetailPrint "[3/4] Downloading the essential model weights ..."
  System::Call 'kernel32::SetEnvironmentVariable(t "PYTHONUNBUFFERED", t "1")'
  nsExec::ExecToLog '"$INSTDIR\resources\uv.exe" run --no-project --python 3.11 "$INSTDIR\resources\src\video_upscaler\desktop\provision.py" --data-dir "$DataDir" --resources-dir "$INSTDIR\resources" --tier essential --tensorrt $TensorrtChoice'
  Pop $0
  IntCmp $0 0 clarity_provisioned clarity_failed clarity_failed

  clarity_provisioned:
    ; 4/4 is the marker, and provision.py writes it — never this script. A hook
    ; that writes it unconditionally is how a failed install used to look
    ; complete, and how the app then skipped provisioning forever.
    DetailPrint "[4/4] Clarity AI engine ready in $DataDir"
    Goto clarity_hook_done

  clarity_failed:
    DetailPrint "[ERROR] The AI engine could not be installed (code $0)."
    DetailPrint "        Log: $DataDir\logs\provision.log"
    MessageBox MB_OK|MB_ICONSTOP "Clarity was installed, but its AI engine could not be set up (code $0).$\n$\nClarity will finish the setup when you next start it.$\nLog: $DataDir\logs\provision.log"
  clarity_hook_done:
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  DetailPrint "Removing Clarity..."
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  ; An update runs the uninstaller before installing the new files. Deleting the
  ; engine there would turn every update into a multi-gigabyte re-download.
  ${If} $UpdateMode = 1
    DetailPrint "Update mode: keeping the AI engine and models."
    Goto clarity_uninstall_done
  ${EndIf}

  ; Regenerable data. An installer can fetch all of this again, and leaving it
  ; behind is precisely the bloat the single-directory layout is meant to avoid.
  RMDir /r "$INSTDIR\python"
  RMDir /r "$INSTDIR\env"
  RMDir /r "$INSTDIR\models"
  RMDir /r "$INSTDIR\tools"
  RMDir /r "$INSTDIR\.cache"
  RMDir /r "$INSTDIR\logs"
  Delete "$INSTDIR\setup.json"
  Delete "$INSTDIR\.setup_complete"
  Delete "$INSTDIR\.provisioning.lock"
  Delete "$INSTDIR\.clarity-write-test"

  ; User media is not ours to delete silently.
  ${If} $DeleteAppDataCheckboxState = 1
    RMDir /r "$INSTDIR\input"
    RMDir /r "$INSTDIR\output"
    RMDir /r "$LOCALAPPDATA\Clarity"
  ${Else}
    MessageBox MB_YESNO|MB_ICONQUESTION "Delete your Clarity videos as well?$\n$\n$INSTDIR\input$INSTDIR\output" IDNO clarity_keep_media
      RMDir /r "$INSTDIR\input"
      RMDir /r "$INSTDIR\output"
    clarity_keep_media:
  ${EndIf}

  RMDir "$INSTDIR"
  DetailPrint "Clarity data removed."
  clarity_uninstall_done:
!macroend
