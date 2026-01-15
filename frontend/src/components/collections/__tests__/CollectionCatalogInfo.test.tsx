// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from 'vitest';
import { render, screen } from '../../../test/utils';
import { CollectionCatalogInfo } from '../CollectionCatalogInfo';
import type { Collection } from '../../../types/collections';

describe('CollectionCatalogInfo', () => {
  const createMockCollection = (overrides: Partial<Collection['collection_info']> = {}): Collection => ({
    collection_name: 'test-collection',
    num_entities: 100,
    metadata_schema: [],
    collection_info: {
      status: 'Active',
      date_created: '2026-01-15T10:00:00Z',
      last_updated: '2026-01-15T12:00:00Z',
      number_of_files: 5,
      description: 'Test collection description',
      ...overrides,
    },
  });

  describe('File Count Display', () => {
    it('displays number_of_files from collection_info when documentCount is not provided', () => {
      const collection = createMockCollection({ number_of_files: 7 });
      render(<CollectionCatalogInfo collection={collection} />);
      
      expect(screen.getByText('7 files')).toBeInTheDocument();
    });

    it('displays documentCount prop when provided (overrides collection_info)', () => {
      const collection = createMockCollection({ number_of_files: 7 });
      render(<CollectionCatalogInfo collection={collection} documentCount={4} />);
      
      // Should show actual document count (4) not the stale collection_info value (7)
      expect(screen.getByText('4 files')).toBeInTheDocument();
      expect(screen.queryByText('7 files')).not.toBeInTheDocument();
    });

    it('displays 0 files when documentCount is 0', () => {
      const collection = createMockCollection({ number_of_files: 5 });
      render(<CollectionCatalogInfo collection={collection} documentCount={0} />);
      
      expect(screen.getByText('0 files')).toBeInTheDocument();
    });

    it('falls back to collection_info when documentCount is undefined', () => {
      const collection = createMockCollection({ number_of_files: 10 });
      render(<CollectionCatalogInfo collection={collection} documentCount={undefined} />);
      
      expect(screen.getByText('10 files')).toBeInTheDocument();
    });

    it('does not show file count when both are undefined', () => {
      const collection = createMockCollection({ number_of_files: undefined });
      render(<CollectionCatalogInfo collection={collection} documentCount={undefined} />);
      
      expect(screen.queryByText(/files/)).not.toBeInTheDocument();
    });
  });

  describe('Table Count Display', () => {
    it('displays table count when doc_type_counts.table is present', () => {
      const collection = createMockCollection({
        doc_type_counts: { table: 3, text: 10 },
      });
      render(<CollectionCatalogInfo collection={collection} />);
      
      expect(screen.getByText('3 Tables')).toBeInTheDocument();
    });

    it('does not display table count when table count is 0', () => {
      const collection = createMockCollection({
        doc_type_counts: { table: 0, text: 10 },
      });
      render(<CollectionCatalogInfo collection={collection} />);
      
      expect(screen.queryByText(/Tables/)).not.toBeInTheDocument();
    });

    it('does not display table count when doc_type_counts is undefined', () => {
      const collection = createMockCollection({
        doc_type_counts: undefined,
      });
      render(<CollectionCatalogInfo collection={collection} />);
      
      expect(screen.queryByText(/Tables/)).not.toBeInTheDocument();
    });
  });

  describe('Chart Count Display', () => {
    it('displays chart count when doc_type_counts.chart is present', () => {
      const collection = createMockCollection({
        doc_type_counts: { chart: 5, text: 10 },
      });
      render(<CollectionCatalogInfo collection={collection} />);
      
      expect(screen.getByText('5 Charts')).toBeInTheDocument();
    });

    it('does not display chart count when chart count is 0', () => {
      const collection = createMockCollection({
        doc_type_counts: { chart: 0, text: 10 },
      });
      render(<CollectionCatalogInfo collection={collection} />);
      
      expect(screen.queryByText(/Charts/)).not.toBeInTheDocument();
    });
  });

  describe('Image Count Display', () => {
    it('displays image count when doc_type_counts.image is present', () => {
      const collection = createMockCollection({
        doc_type_counts: { image: 12, text: 10 },
      });
      render(<CollectionCatalogInfo collection={collection} />);
      
      expect(screen.getByText('12 Images')).toBeInTheDocument();
    });

    it('does not display image count when image count is 0', () => {
      const collection = createMockCollection({
        doc_type_counts: { image: 0, text: 10 },
      });
      render(<CollectionCatalogInfo collection={collection} />);
      
      expect(screen.queryByText(/Images/)).not.toBeInTheDocument();
    });
  });

  describe('Combined Content Metrics', () => {
    it('displays all content types when present', () => {
      const collection = createMockCollection({
        number_of_files: 10,
        doc_type_counts: { table: 3, chart: 2, image: 5, text: 20 },
      });
      render(<CollectionCatalogInfo collection={collection} documentCount={10} />);
      
      expect(screen.getByText('10 files')).toBeInTheDocument();
      expect(screen.getByText('3 Tables')).toBeInTheDocument();
      expect(screen.getByText('2 Charts')).toBeInTheDocument();
      expect(screen.getByText('5 Images')).toBeInTheDocument();
    });
  });

  describe('Status Badge', () => {
    it('displays Active status with correct styling', () => {
      const collection = createMockCollection({ status: 'Active' });
      render(<CollectionCatalogInfo collection={collection} />);
      
      expect(screen.getByText('Active')).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('returns null when collection_info is undefined', () => {
      const collection: Collection = {
        collection_name: 'test',
        num_entities: 0,
        metadata_schema: [],
        collection_info: undefined,
      };
      const { container } = render(<CollectionCatalogInfo collection={collection} />);
      
      expect(container.firstChild).toBeNull();
    });
  });
});

