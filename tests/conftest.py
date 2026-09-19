"""테스트 공통 설정.

- 프로젝트 루트를 import 경로에 넣고, pandas/streamlit보다 먼저 `_pyarrow_compat`을
  불러온다(이 컴퓨터의 Windows 정책이 pyarrow.compute를 막아서 이게 없으면 import
  pandas부터 실패한다).
- 테스트가 실수로 **진짜 Gist(개인 보유종목이 든 저장소)를 읽거나 쓰지 못하게**,
  data.fetch를 import하기 전에 시크릿 파일 경로를 존재하지 않는 곳으로 돌린다.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["STOCK_TIMING_SECRETS_PATH"] = str(ROOT / "tests" / "_no_such_secrets.toml")

import _pyarrow_compat  # noqa: E402,F401
