import { describe, it, expect } from 'vitest';
import { render, screen } from '../../../test/utils';
import { CitationVisualContent } from '../CitationVisualContent';

describe('CitationVisualContent', () => {
  describe('Image Rendering', () => {
    it('renders image with correct src attribute', () => {
      render(
        <CitationVisualContent 
          imageData="base64data123" 
          documentType="image" 
        />
      );
      
      const image = screen.getByRole('img');
      expect(image).toHaveAttribute('src', 'data:image/png;base64,base64data123');
    });

    it('renders image with correct alt text', () => {
      render(
        <CitationVisualContent 
          imageData="imagedata" 
          documentType="screenshot" 
        />
      );
      
      const image = screen.getByRole('img');
      expect(image).toHaveAttribute('alt', 'Citation screenshot');
    });

    it('handles different document types in alt text', () => {
      render(
        <CitationVisualContent 
          imageData="data" 
          documentType="pdf" 
        />
      );
      
      const image = screen.getByRole('img');
      expect(image).toHaveAttribute('alt', 'Citation pdf');
    });

    it('handles different image data', () => {
      render(
        <CitationVisualContent 
          imageData="differentdata456" 
          documentType="image" 
        />
      );
      
      const image = screen.getByRole('img');
      expect(image).toHaveAttribute('src', 'data:image/png;base64,differentdata456');
    });

    it('renders image element', () => {
      render(
        <CitationVisualContent 
          imageData="test" 
          documentType="image" 
        />
      );
      
      expect(screen.getByRole('img')).toBeInTheDocument();
    });
  });

  describe('Data URI Format', () => {
    it('formats data URI correctly with base64 prefix', () => {
      render(
        <CitationVisualContent 
          imageData="testdata" 
          documentType="chart" 
        />
      );
      
      const image = screen.getByRole('img');
      const src = image.getAttribute('src');
      expect(src).toMatch(/^data:image\/png;base64,/);
    });

    it('includes the provided image data in URI', () => {
      render(
        <CitationVisualContent 
          imageData="specificdata789" 
          documentType="diagram" 
        />
      );
      
      const image = screen.getByRole('img');
      expect(image).toHaveAttribute('src', 'data:image/png;base64,specificdata789');
    });

    it('handles empty image data', () => {
      render(
        <CitationVisualContent 
          imageData="" 
          documentType="image" 
        />
      );
      
      const image = screen.getByRole('img');
      expect(image).toHaveAttribute('src', 'data:image/png;base64,');
    });
  });
}); 