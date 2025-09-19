#!/usr/bin/env python3
"""
조용한 음성 인식 실행기 - ALSA 및 gRPC 메시지 필터링
"""

import subprocess
import sys
import os
import threading
import re

def filter_output(process, stream_name):
    """출력 스트림 필터링"""
    stream = getattr(process, stream_name)

    # 필터링할 패턴들
    skip_patterns = [
        r'ALSA lib',
        r'GRPC',
        r'ALTS creds ignored',
        r'All log messages before absl::InitializeLog',
        r'WARNING: All log messages',
        r'E0000.*alts_credentials',
        r'Unknown PCM',
        r'Cannot.*card',
        r'Unable to find definition',
        r'function.*returned error',
        r'Evaluate error',
        r'Invalid field card',
        r'Cannot open device /dev/dsp',
        r'dmix plugin supports only playback',
        r'unable to open slave'
    ]

    for line in iter(stream.readline, b''):
        line_str = line.decode('utf-8', errors='ignore').strip()

        # 필터링 패턴에 매치되지 않으면 출력
        should_skip = any(re.search(pattern, line_str, re.IGNORECASE) for pattern in skip_patterns)

        if not should_skip and line_str:
            if stream_name == 'stderr':
                print(line_str, file=sys.stderr)
            else:
                print(line_str)

def main():
    """음성 인식 프로그램을 조용히 실행"""

    # 환경변수 설정
    env = os.environ.copy()
    env.update({
        'ALSA_PCM_CARD': 'hw:0',
        'ALSA_PCM_DEVICE': '0',
        'GRPC_VERBOSITY': 'NONE',
        'GRPC_LOG_SEVERITY_LEVEL': 'ERROR'
    })

    # voice_recognition_improved.py 실행
    cmd = [sys.executable, 'voice_recognition_improved.py']

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,
            universal_newlines=False
        )

        # 출력 스트림 필터링 스레드 시작
        stdout_thread = threading.Thread(
            target=filter_output,
            args=(process, 'stdout'),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=filter_output,
            args=(process, 'stderr'),
            daemon=True
        )

        stdout_thread.start()
        stderr_thread.start()

        # 프로세스 완료 대기
        process.wait()

    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
        process.terminate()
        process.wait()
    except Exception as e:
        print(f"실행 중 오류 발생: {e}", file=sys.stderr)
        return 1

    return process.returncode

if __name__ == "__main__":
    sys.exit(main())