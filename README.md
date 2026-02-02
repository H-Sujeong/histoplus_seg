# 🧩 Execution Environment & `run.sh` Usage Guide

*(with VS Code Dev Containers)*

> 본 프로젝트는 **Docker 기반 실행 환경**을 사용하며
>
> * **배치/추론 실행**은 `run.sh`
> * **VS Code 기반 개발/디버깅**은 *Dev Containers*
>   로 **용도를 명확히 분리**하여 운영합니다.

---

## 🔟 VS Code Dev Containers 개요

VS Code Dev Containers는
**Docker 컨테이너를 “개발 환경”으로 직접 사용하는 기능**입니다.

본 프로젝트에서는 Dev Containers를 통해:

* 로컬 Python/Conda 환경 오염 없이 개발
* 서버 이동 시에도 **동일한 개발 환경 유지**
* Jupyter / Python Extension / LSP 안정적 사용
* `run.sh`와 **동일한 Docker 이미지 기반 개발**

을 목표로 합니다.

> ⚠️ Dev Containers는 **실험 실행(run.sh)** 을 대체하지 않습니다.
> 👉 *개발/디버깅 전용*입니다.

---

## 1️⃣1️⃣ Dev Containers를 사용하는 이유

### ❓ 왜 `docker exec` / `attach`가 아닌가?

* `docker attach` / `exec`는 **root로 접속되는 경우가 많음**
* VS Code Server, Python Extension, Jupyter 커널 경로가 꼬이기 쉬움
* UID/GID remapping이 불안정

👉 **Dev Containers는 이 문제를 구조적으로 해결**합니다.

---

## 1️⃣2️⃣ Dev Containers 구성 파일 위치

프로젝트 루트에 다음 구조가 존재합니다.

```text
.histoplus_seg/
├── .devcontainer/
│   ├── devcontainer.json
│   └── (선택) dev.Dockerfile
├── run.sh
├── Dockerfile
└── ...
```

* **`devcontainer.json`**
  → VS Code가 컨테이너를 어떻게 띄울지 정의
* (선택) `dev.Dockerfile`
  → base 이미지 위에 *dev 전용 설정*을 얹고 싶을 때 사용

---

## 1️⃣3️⃣ Dev Containers에서 사용하는 Docker 이미지

Dev Containers는 다음 이미지를 사용합니다.

```json
"image": "hist:base-dev"
```

### 🔹 `hist:base` vs `hist:base-dev`

| 이미지             | 용도                           |
| --------------- | ---------------------------- |
| `hist:base`     | 배치/추론 실행 (run.sh)            |
| `hist:base-dev` | VS Code 개발용 (Dev Containers) |

`hist:base-dev`는 다음을 보장합니다.

* placeholder `appuser` 존재
* entrypoint 기반 UID/GID remap 가능
* VS Code가 `remoteUser=appuser`로 정상 실행 가능

---

## 1️⃣4️⃣ devcontainer.json 핵심 설정 설명

### 📌 사용자 관련

```json
"remoteUser": "appuser",
"containerUser": "appuser"
```

* VS Code Server / 터미널 / Jupyter 모두 `appuser`로 실행
* root로 들어가는 문제 방지

---

### 📌 Python / Jupyter 설정

```json
"python.defaultInterpreterPath": "/opt/micromamba/envs/hist/bin/python",
"jupyter.jupyterServerType": "local",
"jupyter.kernelspecPaths": [
  "/opt/micromamba/envs/hist/share/jupyter/kernels",
  "/home/appuser/.local/share/jupyter/kernels"
]
```

이를 통해:

* VS Code Python Extension이 **hist env를 기본 인터프리터로 인식**
* Jupyter 커널 자동 탐색
* 커널이 안 뜨는 문제 방지

---

### 📌 GPU / IPC / Mount 설정

Dev Containers는 `run.sh`와 동일한 실행 조건을 유지합니다.

```json
"runArgs": [
  "--gpus", "all",
  "--ipc=host"
]
```

```json
"mounts": [
  "source=/home,target=/home,type=bind",
  "source=/data,target=/data,type=bind",
  "source=/home/nas2_fast,target=/home/nas2_fast,type=bind"
]
```

👉 **코드 경로 수정 없이** run.sh ↔ VS Code 환경 전환 가능

---

## 1️⃣5️⃣ “Reopen in Container” 동작 방식 (중요)

VS Code에서
**`Reopen in Container`** 를 실행하면 다음 순서로 동작합니다.

### 🔄 내부 동작 순서

1. `.devcontainer/devcontainer.json` 탐색
2. 지정된 Docker 이미지 확인 (`hist:base-dev`)
3. 필요 시 **Dev Containers 전용 파생 컨테이너 생성**

   * 이름: `vsc-<project>-<hash>`
4. 컨테이너 실행 시:

   * `remoteUser=appuser`
   * UID/GID 자동 보정 (VS Code 기능)
5. VS Code Server 설치 (`~/.vscode-server`)
6. Python/Jupyter 확장 로딩

👉 **이 컨테이너는 run.sh로 띄운 컨테이너와 “별개”입니다.**

---

## 1️⃣6️⃣ Reopen 메뉴 옵션 설명

VS Code Command Palette (`Ctrl+Shift+P`) → `Dev Containers`

| 메뉴                              | 의미                    |
| ------------------------------- | --------------------- |
| **Reopen in Container**         | 처음 컨테이너 열기            |
| **Rebuild and Reopen**          | 이미지/설정 변경 후 강제 재빌드    |
| **Reopen Locally**              | 컨테이너 종료 후 로컬로 복귀      |
| **Attach to Running Container** | ❌ 비권장 (root 문제 발생 가능) |

> ⚠️ **`Attach to Running Container`는 권장하지 않습니다.**
> run.sh로 띄운 컨테이너에 attach 시 root로 접속될 수 있으며,
> 이는 Dev Containers 설계와 다릅니다.

---

## 1️⃣7️⃣ Dev Containers 권장 사용 패턴

### ✅ 권장

* 코드 작성 / 디버깅
* Jupyter Notebook 작업
* Python LSP / 자동완성
* 커널 기반 실험

👉 **VS Code → Reopen in Container**

---

### ❌ 비권장

* 장시간 배치 추론
* 멀티 실험 병렬 실행
* 대규모 데이터 처리

👉 이 경우 **`run.sh` 사용**

---

## 1️⃣8️⃣ run.sh vs Dev Containers 역할 분리 요약

| 항목     | run.sh           | Dev Containers |
| ------ | ---------------- | -------------- |
| 목적     | 배치/추론 실행         | 개발/디버깅         |
| 컨테이너   | 직접 실행            | VS Code 관리     |
| 사용자    | entrypoint remap | remoteUser     |
| Attach | X                | O              |
| 안정성    | 높음               | 높음             |
| 병렬 실행  | O                | X              |

---

## 📌 최종 요약

* **run.sh**

  * 실험/추론 실행 표준
  * 서버 자원 사용에 최적
* **VS Code Dev Containers**

  * 개발/디버깅 전용
  * Python/Jupyter/권한 문제 최소화
* 두 환경은 **의도적으로 분리**되어 있으며
  이는 설계상 정상입니다.

