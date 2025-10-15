import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '../../../test/utils';
import FileUploaderWithMetadata from '../FileUploaderWithMetadata';

// Mock child components
vi.mock('../FileInput', () => ({
  FileInput: () => <div data-testid="file-input">File Input</div>
}));

vi.mock('../FilesList', () => ({
  FilesList: () => <div data-testid="files-list">Files List</div>
}));

describe('FileUploaderWithMetadata', () => {
  describe('Component Composition', () => {
    it('renders FileInput component', () => {
      render(<FileUploaderWithMetadata />);
      expect(screen.getByTestId('file-input')).toBeInTheDocument();
    });

    it('renders FilesList component', () => {
      render(<FileUploaderWithMetadata />);
      expect(screen.getByTestId('files-list')).toBeInTheDocument();
    });
  });
}); 