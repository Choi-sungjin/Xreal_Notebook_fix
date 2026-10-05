# XREAL One Pro × 웹캠 머리 방향 센서 퓨전

노트북 웹캠으로 안경 쓴 얼굴을 추적하고(MediaPipe Face Landmarker), XREAL One Pro 글래스의
자이로(1000 Hz)와 융합해 **좌석(카메라) 기준 머리 방향**을 추정합니다. 측정상 가장 큰 효과는
떨림 감소(카메라 대비 약 4–7배)이고, 설계상 출력은 최신 자이로까지 적분하므로 카메라 지연(~40 ms)을
따르지 않습니다(지연 자체는 측정하지 않았음). 움직이는 탈것 안에서도 탈것의 회전은
바이어스로 흡수되어 머리의 상대 움직임만 남습니다.

**글래스를 벗었다 쓰거나 위치를 고친 뒤에는 재캘리브레이션하세요** (실행 중 `c` 키). 세션 간 축 정합이 수 도 달라질 수 있습니다.

## exe로 실행 (Windows)

```
powershell -ExecutionPolicy Bypass -File build_exe.ps1
```

`%LOCALAPPDATA%\Programs\XrealHeadFusion\XrealHeadFusion.exe`로 설치되고 바탕화면에 **XREAL Head Fusion** 바로가기가 생깁니다.
더블클릭하면 캘리브레이션 파일이 없을 때는 캘리브레이션부터, 있으면 바로 실시간 융합을 시작합니다.
캘리브레이션은 `%APPDATA%\XrealHeadFusion\calib.json`에 저장됩니다. 빌드용 가상환경과 중간 파일은
`%LOCALAPPDATA%\XrealHeadFusion`에 만들어져 프로젝트 폴더(클라우드 동기화 폴더일 수 있음)에는 쌓이지 않습니다.

실행 중 키: `q` 종료 · `z` 지금 자세를 정면으로 지정 · `c` 재캘리브레이션 · `d` 카메라 보정 끄기(테스트)

## 정면 재보정

웹캠 자세는 사용자가 카메라를 정면으로 보고 멈춰 있을 때 가장 정확합니다(약 0.5°). 그래서 정면(캘리브레이션 때
정지 자세 기준 15° 이내, `z`로 다시 지정 가능)에서 0.17초 이상 멈춰 있으면 카메라 보정을 강하게 걸고,
20° 넘게 틀어져 있으면 즉시 카메라 값으로 맞춥니다. 크게 움직이는 동안 쌓인 오차는 다시 정면을 보는 순간 사라집니다.
녹화본에 10° 오차를 넣어 본 시험에서 2° 이내 복귀가 1.10초 → 0.51초로 빨라졌습니다(정면 정지 구간 1회 기준).

## 소스로 실행

```
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

`models/face_landmarker.task`(Google MediaPipe 공식 모델, Apache-2.0)가 필요합니다. `build_exe.ps1`이 없으면 내려받습니다.
(수동: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`)

## 사용 순서

```
.venv/Scripts/python.exe tracker.py imu-test            # 1. 글래스 IMU 수신 확인
.venv/Scripts/python.exe tracker.py face-test           # 2. 웹캠 얼굴 자세 확인
.venv/Scripts/python.exe tracker.py calibrate           # 3. 화면 안내대로 고개 움직이기 (약 25초) → calib.json
.venv/Scripts/python.exe tracker.py run                 # 4. 실시간 융합
```

- `run --duration 45 --dropout 20-30 --log out.csv --record rec.npz` : 테스트용 (중간 10초 카메라 보정 끄기)
- `replay rec.npz --dropout 20-30 [--kp 0.06 --ki 0.3 --rate-ref 0.6 --bias-gate 0.25 --home-deg 15 --home-kp 0.25]` : 녹화본으로 오프라인 재생·튜닝
- `calibrate --from-raw` : 마지막 캘리브레이션 원시 데이터(`calib_raw.npz`)로 다시 계산

## 구조

| 파일 | 역할 |
|---|---|
| `xreal_imu.py` | 글래스 USB 네트워크(169.254.2.1:52998)에서 IMU 패킷 수신·파싱, 호스트 시계로 정렬 |
| `head_pose.py` | 웹캠 캡처, 가장 큰 얼굴만 골라 머리 회전 추정 (뒤 사람 무시) |
| `fusion.py` | 카메라↔IMU 캘리브레이션(지연·축 정합), 지연 보상 상보 필터(Mahony PI) |
| `tracker.py` | CLI·exe 진입점, 화면 표시, 로그·녹화·재생, 결과 요약 |
| `build_exe.ps1` | PyInstaller 빌드, 사용자 설치, 바탕화면 바로가기 |
| `so3.py` | 회전 유틸 |

IMU 패킷 형식은 공식 SDK가 아니라 스트림을 분석해 알아낸 것이라 펌웨어 업데이트로 바뀔 수 있습니다.

## 측정 결과 (노트북 앞에 앉은 상태)

| 항목 | 결과 |
|---|---|
| IMU | 1000 Hz, 패킷 손실 0, 자이로 단위 rad/s 확인 |
| 웹캠 추적 | 선글라스 착용 상태 얼굴 검출 99–100%, 22–28 fps |
| 카메라 지연 (캘리브레이션) | 약 40 ms (융합 시 보상) |
| **멈춰 있을 때 떨림** | **카메라 0.75–0.86° → 융합 0.10–0.20° / 프레임** |
| 멈춘 순간 정확도 (카메라 대비) | 중앙값 0.75–1.6° |
| 카메라 보정 끄고 IMU만 — 가만히 있을 때 10초 | 0.7° → 0.3° (최대 2.0°) |
| 카메라 보정 끄고 IMU만 — 크게 움직일 때 6초 (pitch·roll 약 33°) | 3.8° → 1.4°, 멈춘 순간 약 4° · 보정 안 한 자이로 +7° |
| 카메라 보정 끄고 IMU만 — 크게 움직일 때 10초 (roll 71°) | 0.0° → 8.8° (최대 12–16°) · 보정 안 한 자이로 +13° |
| 보정 안 한 자이로만 | 약 70–90°/분 드리프트 (글래스 자이로 바이어스 + 탈것 회전) |

얼굴을 놓쳤을 때 IMU로 버티는 효과는 **몇 초 단위**에서 확실합니다. 크게 움직이며 10초가 지나면
보정 안 한 자이로보다 약 30% 나은 정도라, 그보다 긴 공백에는 적합하지 않습니다.

**한계:**
- 고개를 크게 기울이거나(갸웃 30° 이상) 빠르게 돌리면 웹캠 얼굴 자세 자체가 부정확해져,
  카메라를 기준으로 하는 평가가 2–6° 수준 이상을 구분하지 못합니다.
- 두 세션에서 구한 캘리브레이션이 7° 달랐는데(각각의 부트스트랩 95% 범위 4–5°보다 큼) 서로 안 본
  데이터에서 성능은 같았습니다. 즉 세션 간 축 정합 불확실성은 약 ±5–7°이고, 지금 지표로는 어느 쪽이
  맞는지 가릴 수 없습니다. 글래스가 얼굴에서 밀렸을 가능성도 있어 착용할 때마다 재캘리브레이션을 권장합니다.
- 더 줄이려면 정답 기준(예: 마커나 다른 트래커)이 있는 평가, 또는 더 정확한 머리 자세 추정이 필요합니다.
