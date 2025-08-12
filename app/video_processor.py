import cv2
import numpy as np
import logging
import tempfile
import os
from pathlib import Path
from typing import List, Optional
import time

logger = logging.getLogger(__name__)

class VideoProcessor:
    """비디오 전처리를 담당하는 클래스"""

    def __init__(self, clip_duration: float = 2.0, stride: float = 1.0, num_frames: int = 32):
        self.clip_duration = clip_duration
        self.stride = stride
        self.num_frames = num_frames
        logger.info(f"VideoProcessor 초기화 - 클립 길이: {clip_duration}초, 스트라이드: {stride}초, 프레임 수: {num_frames}")

    def preprocess_video(self, video_bytes: bytes) -> List[List[np.ndarray]]:
        """비디오 바이트를 클립 리스트로 전처리"""
        tmp_file_path = None
        try:
            # 임시 파일 생성 (Windows 호환성을 위해 컨텍스트 매니저 밖에서 처리)
            tmp_fd, tmp_file_path = tempfile.mkstemp(suffix='.mp4')

            # 파일 디스크립터를 통해 데이터 쓰기
            with os.fdopen(tmp_fd, 'wb') as tmp_file:
                tmp_file.write(video_bytes)
                tmp_file.flush()

            # VideoCapture 열기
            cap = cv2.VideoCapture(tmp_file_path)

            if not cap.isOpened():
                raise ValueError("비디오 파일을 열 수 없습니다")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            logger.info(f"비디오 정보 - FPS: {fps}, 총 프레임: {total_frames}, 길이: {duration:.2f}초")

            # 슬라이딩 윈도우로 클립 추출
            clips_list = []
            start_time = 0.0

            while start_time + self.clip_duration <= duration:
                clip_frames = self._extract_frames_for_vivit(cap, start_time, self.clip_duration, fps)
                if clip_frames is not None:
                    clips_list.append(clip_frames)
                start_time += self.stride

            # VideoCapture 완전히 해제
            cap.release()
            cv2.destroyAllWindows()

            if not clips_list:
                raise ValueError("비디오에서 클립을 추출할 수 없습니다")

            logger.info(f"추출된 클립 수: {len(clips_list)}")
            return clips_list

        except Exception as e:
            logger.error(f"비디오 전처리 실패: {e}")
            raise
        finally:
            # 임시 파일 정리 (에러가 발생해도 실행)
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    # 약간의 지연을 주어 파일 핸들이 완전히 해제되도록 함
                    time.sleep(0.1)
                    os.unlink(tmp_file_path)
                    logger.debug(f"임시 파일 삭제 완료: {tmp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"임시 파일 삭제 실패: {cleanup_error}, 경로: {tmp_file_path}")

    def _extract_frames_for_vivit(self, cap, start_time: float, duration: float, fps: float) -> Optional[List[np.ndarray]]:
        """ViViT용 프레임 추출 (지정된 프레임 수만큼)"""
        try:
            start_frame = int(start_time * fps)
            end_frame = int((start_time + duration) * fps)
            total_clip_frames = end_frame - start_frame

            if total_clip_frames < self.num_frames:
                logger.warning(f"클립 프레임 수가 부족합니다: {total_clip_frames}")
                return None

            # 정확한 프레임 균등 샘플링
            if total_clip_frames == self.num_frames:
                # 정확히 필요한 프레임 수면 모든 프레임 사용
                frame_indices = list(range(start_frame, end_frame))
            else:
                # 균등 간격으로 선택 (소수점 계산 후 정수 변환)
                step = (total_clip_frames - 1) / (self.num_frames - 1)
                frame_indices = [start_frame + int(i * step) for i in range(self.num_frames)]

                # 중복 제거 및 정렬
                frame_indices = sorted(list(set(frame_indices)))

                # 정확히 필요한 프레임 수가 되도록 조정
                while len(frame_indices) < self.num_frames:
                    # 빈 곳에 프레임 추가
                    for i in range(len(frame_indices) - 1):
                        if frame_indices[i+1] - frame_indices[i] > 1:
                            frame_indices.insert(i+1, frame_indices[i] + 1)
                            break
                    else:
                        # 마지막에 추가
                        if frame_indices[-1] < end_frame - 1:
                            frame_indices.append(frame_indices[-1] + 1)
                        else:
                            break

                # 정확히 필요한 프레임 수로 자르기
                frame_indices = frame_indices[:self.num_frames]

            logger.debug(f"선택된 프레임 인덱스: {frame_indices[:5]}...{frame_indices[-5:]} (총 {len(frame_indices)}개)")

            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"프레임 {frame_idx} 읽기 실패")
                    continue

                # BGR to RGB 변환 및 크기 조정
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

                # 0-255 범위를 0-1로 정규화 (명시적 타입 지정)
                frame_normalized = frame_resized.astype(np.float32) / 255.0

                frames.append(frame_normalized)

            if len(frames) != self.num_frames:
                logger.warning(f"추출된 프레임 수가 {self.num_frames}개가 아닙니다: {len(frames)}")
                return None

            logger.debug(f"프레임 추출 완료: {len(frames)}개, 형태: {frames[0].shape}")
            return frames

        except Exception as e:
            logger.error(f"프레임 추출 실패: {e}")
            return None

    def _extract_original_frames_for_saving(self, cap, start_time: float, duration: float, fps: float) -> Optional[List[np.ndarray]]:
        """원본 해상도 프레임 추출 (저장용)"""
        try:
            start_frame = int(start_time * fps)
            end_frame = int((start_time + duration) * fps)
            total_clip_frames = end_frame - start_frame

            if total_clip_frames < self.num_frames:
                logger.warning(f"클립 프레임 수가 부족합니다: {total_clip_frames}")
                return None

            # 프레임 균등 샘플링을 위한 인덱스 계산
            frame_indices = np.linspace(start_frame, end_frame - 1, self.num_frames, dtype=int)

            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"프레임 {frame_idx} 읽기 실패")
                    continue

                # BGR to RGB 변환만 (리사이징과 정규화 없음)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            if len(frames) != self.num_frames:
                logger.warning(f"추출된 프레임 수가 {self.num_frames}개가 아닙니다: {len(frames)}")
                return None

            return frames

        except Exception as e:
            logger.error(f"원본 프레임 추출 실패: {e}")
            return None

    def save_clips_as_videos(self, clips_list: List[List[np.ndarray]], audit_id: str, output_dir: str = "output") -> List[str]:
        """샘플링된 클립들을 mp4 파일로 저장"""
        saved_paths = []

        try:
            # output 디렉토리 생성
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            for i, clip_frames in enumerate(clips_list):
                # 파일명 생성
                video_filename = f"{audit_id}_clip_{i+1:02d}.mp4"
                video_path = output_path / video_filename

                # VideoWriter 설정
                height, width = clip_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 16  # 프레임 수를 2초로 재생하기 위한 fps 계산

                out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

                if not out.isOpened():
                    logger.error(f"VideoWriter 초기화 실패: {video_path}")
                    continue

                # 프레임 쓰기
                for frame in clip_frames:
                    # 정규화된 프레임을 0-255 범위로 변환
                    frame_uint8 = (frame * 255).astype(np.uint8)
                    # RGB를 BGR로 변환 (OpenCV 요구사항)
                    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)

                out.release()
                saved_paths.append(str(video_path))
                logger.info(f"클립 {i+1} 저장 완료: {video_path}")

            return saved_paths

        except Exception as e:
            logger.error(f"비디오 클립 저장 실패: {e}")
            return saved_paths
