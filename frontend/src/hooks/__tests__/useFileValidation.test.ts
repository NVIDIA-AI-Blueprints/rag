import { describe, it, expect } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useFileValidation } from '../useFileValidation';

describe('useFileValidation', () => {
  const defaultOptions = {
    acceptedTypes: ['.pdf', '.txt', '.mp3', '.wav'],
    maxFileSize: 400 // 400MB for all file types
  };

  describe('validateFile', () => {
    it('should accept valid non-audio files within size limit', () => {
      const { result } = renderHook(() => useFileValidation(defaultOptions));
      
      // 10MB PDF file
      const pdfFile = new File(['x'.repeat(10 * 1024 * 1024)], 'document.pdf', { type: 'application/pdf' });
      const error = result.current.validateFile(pdfFile);
      
      expect(error).toBeNull();
    });

    it('should reject files exceeding size limit', () => {
      const { result } = renderHook(() => useFileValidation(defaultOptions));
      
      // 450MB PDF file (exceeds 400MB limit)
      const largePdfFile = new File(['content'], 'large.pdf', { type: 'application/pdf' });
      Object.defineProperty(largePdfFile, 'size', { value: 450 * 1024 * 1024, writable: false });
      const error = result.current.validateFile(largePdfFile);
      
      expect(error).toBe('File too large. Max size: 400MB');
    });

    it('should accept audio files within size limit', () => {
      const { result } = renderHook(() => useFileValidation(defaultOptions));
      
      // 200MB MP3 file (within 400MB limit)
      const mp3File = new File(['content'], 'audio.mp3', { type: 'audio/mpeg' });
      Object.defineProperty(mp3File, 'size', { value: 200 * 1024 * 1024, writable: false });
      
      const error = result.current.validateFile(mp3File);
      
      expect(error).toBeNull();
    });

    it('should accept WAV files within size limit', () => {
      const { result } = renderHook(() => useFileValidation(defaultOptions));
      
      // 350MB WAV file (within 400MB limit)
      const wavFile = new File(['content'], 'audio.wav', { type: 'audio/wav' });
      Object.defineProperty(wavFile, 'size', { value: 350 * 1024 * 1024, writable: false });

      const error = result.current.validateFile(wavFile);
      
      expect(error).toBeNull();
    });

    it('should reject MP3 files exceeding size limit', () => {
      const { result } = renderHook(() => useFileValidation(defaultOptions));
      
      // 450MB MP3 file (exceeds 400MB limit)
      const largeMp3File = new File(['content'], 'large-audio.mp3', { type: 'audio/mpeg' });
      Object.defineProperty(largeMp3File, 'size', { value: 450 * 1024 * 1024, writable: false });

      const error = result.current.validateFile(largeMp3File);

      expect(error).toBe('File too large. Max size: 400MB');
    });

    it('should reject WAV files exceeding size limit', () => {
      const { result } = renderHook(() => useFileValidation(defaultOptions));
      
      // 450MB WAV file (exceeds 400MB limit)
      const largeWavFile = new File(['content'], 'large-audio.wav', { type: 'audio/wav' });
      Object.defineProperty(largeWavFile, 'size', { value: 450 * 1024 * 1024, writable: false });

      const error = result.current.validateFile(largeWavFile);

      expect(error).toBe('File too large. Max size: 400MB');
    });

    it('should handle case-insensitive file extensions', () => {
      const { result } = renderHook(() => useFileValidation(defaultOptions));
      
      // Test uppercase extensions
      const mp3File = new File(['x'.repeat(10 * 1024 * 1024)], 'audio.MP3', { type: 'audio/mpeg' });
      const wavFile = new File(['x'.repeat(10 * 1024 * 1024)], 'audio.WAV', { type: 'audio/wav' });
      
      expect(result.current.validateFile(mp3File)).toBeNull();
      expect(result.current.validateFile(wavFile)).toBeNull();
    });



    it('should reject unsupported file types', () => {
      const { result } = renderHook(() => useFileValidation(defaultOptions));
      
      const unsupportedFile = new File(['content'], 'document.docx', { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' });
      const error = result.current.validateFile(unsupportedFile);
      
      expect(error).toBe('File type not supported. Accepted: .pdf, .txt, .mp3, .wav');
    });

    it('should handle files without extensions', () => {
      const { result } = renderHook(() => useFileValidation(defaultOptions));
      
      const noExtFile = new File(['content'], 'filename_no_extension', { type: 'text/plain' });
      const error = result.current.validateFile(noExtFile);
      
      expect(error).toBe('File type not supported. Accepted: .pdf, .txt, .mp3, .wav');
    });
  });

  describe('validateFiles', () => {
    it('should validate multiple files correctly', () => {
      const { result } = renderHook(() => useFileValidation(defaultOptions));
      
      const smallPdf = new File(['content'], 'small.pdf', { type: 'application/pdf' });
      Object.defineProperty(smallPdf, 'size', { value: 10 * 1024 * 1024, writable: false });
      
      const largePdf = new File(['content'], 'large.pdf', { type: 'application/pdf' });
      Object.defineProperty(largePdf, 'size', { value: 60 * 1024 * 1024, writable: false });
      
      const audioMp3 = new File(['content'], 'audio.mp3', { type: 'audio/mpeg' });
      Object.defineProperty(audioMp3, 'size', { value: 200 * 1024 * 1024, writable: false });
      
      const hugeAudioWav = new File(['content'], 'huge-audio.wav', { type: 'audio/wav' });
      Object.defineProperty(hugeAudioWav, 'size', { value: 500 * 1024 * 1024, writable: false });
      
      const unsupportedFile = new File(['content'], 'unsupported.docx', { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' });
      
      const files = [smallPdf, largePdf, audioMp3, hugeAudioWav, unsupportedFile];
      
      const { valid, invalid } = result.current.validateFiles(files);
      
      expect(valid).toHaveLength(3);
      expect(valid[0].name).toBe('small.pdf');
      expect(valid[1].name).toBe('large.pdf');
      expect(valid[2].name).toBe('audio.mp3');
      
      expect(invalid).toHaveLength(2);
      expect(invalid[0].error).toBe('File too large. Max size: 400MB');
      expect(invalid[1].error).toBe('File type not supported. Accepted: .pdf, .txt, .mp3, .wav');
    });

    it('should return all valid when all files pass validation', () => {
      const { result } = renderHook(() => useFileValidation(defaultOptions));
      
      const files = [
        new File(['content'], 'document1.pdf', { type: 'application/pdf' }),
        new File(['content'], 'document2.txt', { type: 'text/plain' }),
        new File(['content'], 'audio.mp3', { type: 'audio/mpeg' })
      ];
      
      // Set file sizes using Object.defineProperty to avoid creating massive strings
      Object.defineProperty(files[0], 'size', { value: 10 * 1024 * 1024, writable: false }); // 10MB
      Object.defineProperty(files[1], 'size', { value: 5 * 1024 * 1024, writable: false });  // 5MB
      Object.defineProperty(files[2], 'size', { value: 100 * 1024 * 1024, writable: false }); // 100MB
      
      const { valid, invalid } = result.current.validateFiles(files);
      
      expect(valid).toHaveLength(3);
      expect(invalid).toHaveLength(0);
    });

    it('should return all invalid when all files fail validation', () => {
      const { result } = renderHook(() => useFileValidation(defaultOptions));
      
      const largePdf = new File(['content'], 'large.pdf', { type: 'application/pdf' });
      Object.defineProperty(largePdf, 'size', { value: 450 * 1024 * 1024, writable: false });
      
      const unsupportedFile = new File(['content'], 'unsupported.docx', { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' });
      
      const hugeAudioMp3 = new File(['content'], 'huge-audio.mp3', { type: 'audio/mpeg' });
      Object.defineProperty(hugeAudioMp3, 'size', { value: 500 * 1024 * 1024, writable: false });
      
      const files = [largePdf, unsupportedFile, hugeAudioMp3];
      
      const { valid, invalid } = result.current.validateFiles(files);
      
      expect(valid).toHaveLength(0);
      expect(invalid).toHaveLength(3);
    });
  });

  describe('edge cases', () => {
    it('should handle zero-byte files', () => {
      const { result } = renderHook(() => useFileValidation(defaultOptions));
      
      const emptyFile = new File([], 'empty.txt', { type: 'text/plain' });
      const error = result.current.validateFile(emptyFile);
      
      expect(error).toBeNull(); // Empty files should be valid
    });

    it('should handle exactly at size limits', () => {
      const { result } = renderHook(() => useFileValidation(defaultOptions));
      
      // Exactly 400MB PDF
      const exactSizePdf = new File(['content'], 'exact.pdf', { type: 'application/pdf' });
      Object.defineProperty(exactSizePdf, 'size', { value: 400 * 1024 * 1024, writable: false });
      expect(result.current.validateFile(exactSizePdf)).toBeNull();
      
      // Exactly 400MB audio
      const exactSizeAudio = new File(['content'], 'exact.mp3', { type: 'audio/mpeg' });
      Object.defineProperty(exactSizeAudio, 'size', { value: 400 * 1024 * 1024, writable: false });
      expect(result.current.validateFile(exactSizeAudio)).toBeNull();
      
      // One byte over limit
      const overSizePdf = new File(['content'], 'over.pdf', { type: 'application/pdf' });
      Object.defineProperty(overSizePdf, 'size', { value: 400 * 1024 * 1024 + 1, writable: false });
      expect(result.current.validateFile(overSizePdf)).toBe('File too large. Max size: 400MB');
      
      const overSizeAudio = new File(['content'], 'over.mp3', { type: 'audio/mpeg' });
      Object.defineProperty(overSizeAudio, 'size', { value: 400 * 1024 * 1024 + 1, writable: false });
      expect(result.current.validateFile(overSizeAudio)).toBe('File too large. Max size: 400MB');
    });
  });
});
