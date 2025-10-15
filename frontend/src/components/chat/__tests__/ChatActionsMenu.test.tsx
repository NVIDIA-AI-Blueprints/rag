// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '../../../test/utils';
import userEvent from '@testing-library/user-event';
import { ChatActionsMenu } from '../ChatActionsMenu';
import { useChatStore } from '../../../store/useChatStore';

vi.mock('../../../store/useChatStore');

const mockClearMessages = vi.fn();
const mockMessages = [
  { id: '1', content: 'Hello', role: 'user', timestamp: Date.now() },
  { id: '2', content: 'Hi there!', role: 'assistant', timestamp: Date.now() }
];

describe('ChatActionsMenu', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useChatStore).mockReturnValue({
      messages: mockMessages,
      clearMessages: mockClearMessages,
    } as ReturnType<typeof useChatStore>);
  });

  describe('Menu Toggle', () => {
    it('renders plus button', () => {
      render(<ChatActionsMenu />);
      const button = screen.getByRole('button', { name: /chat options/i });
      expect(button).toBeInTheDocument();
    });

    it('opens menu when plus button is clicked', async () => {
      const user = userEvent.setup();
      render(<ChatActionsMenu />);
      
      const button = screen.getByRole('button', { name: /chat options/i });
      await user.click(button);
      
      expect(screen.getByText('Clear chat')).toBeInTheDocument();
    });

    it('closes menu when clicked outside', async () => {
      const user = userEvent.setup();
      render(<ChatActionsMenu />);
      
      // Open menu
      const button = screen.getByRole('button', { name: /chat options/i });
      await user.click(button);
      expect(screen.getByText('Clear chat')).toBeInTheDocument();
      
      // Click outside
      await user.click(document.body);
      
      await waitFor(() => {
        expect(screen.queryByText('Clear chat')).not.toBeInTheDocument();
      });
    });

    it('closes menu when plus button is clicked again', async () => {
      const user = userEvent.setup();
      render(<ChatActionsMenu />);
      
      const button = screen.getByRole('button', { name: /chat options/i });
      
      // Open menu
      await user.click(button);
      expect(screen.getByText('Clear chat')).toBeInTheDocument();
      
      // Close menu
      await user.click(button);
      await waitFor(() => {
        expect(screen.queryByText('Clear chat')).not.toBeInTheDocument();
      });
    });
  });

  describe('Clear Chat Functionality', () => {
    it('shows enabled clear chat option when messages exist', async () => {
      const user = userEvent.setup();
      render(<ChatActionsMenu />);
      
      const button = screen.getByRole('button', { name: /chat options/i });
      await user.click(button);
      
      const clearButton = screen.getByRole('menuitem', { name: /clear chat/i });
      expect(clearButton).toBeInTheDocument();
      expect(clearButton).not.toHaveAttribute('aria-disabled', 'true');
    });

    it('shows disabled clear chat option when no messages exist', async () => {
      const user = userEvent.setup();
      
      // Mock empty messages
      vi.mocked(useChatStore).mockReturnValue({
        messages: [],
        clearMessages: mockClearMessages,
      } as ReturnType<typeof useChatStore>);
      
      render(<ChatActionsMenu />);
      
      const button = screen.getByRole('button', { name: /chat options/i });
      await user.click(button);
      
      const clearButton = screen.getByRole('menuitem', { name: /clear chat/i });
      expect(clearButton).toBeInTheDocument();
      expect(clearButton).toHaveAttribute('aria-disabled', 'true');
    });

    it('calls clearMessages when confirmed', async () => {
      const user = userEvent.setup();
      
      render(<ChatActionsMenu />);
      
      const button = screen.getByRole('button', { name: /chat options/i });
      await user.click(button);
      
      const clearButton = screen.getByRole('menuitem', { name: /clear chat/i });
      await user.click(clearButton);
      
      // Modal should be visible
      expect(screen.getByText(/Are you sure you want to clear all chat messages/)).toBeInTheDocument();
      
      // Click the confirm button in the modal
      const confirmButton = screen.getByRole('button', { name: /clear chat/i });
      await user.click(confirmButton);
      
      expect(mockClearMessages).toHaveBeenCalled();
    });

    it('does not call clearMessages when cancelled', async () => {
      const user = userEvent.setup();
      
      render(<ChatActionsMenu />);
      
      const button = screen.getByRole('button', { name: /chat options/i });
      await user.click(button);
      
      const clearButton = screen.getByRole('menuitem', { name: /clear chat/i });
      await user.click(clearButton);
      
      // Modal should be visible
      expect(screen.getByText(/Are you sure you want to clear all chat messages/)).toBeInTheDocument();
      
      // Click the cancel button in the modal
      const cancelButton = screen.getByRole('button', { name: /cancel/i });
      await user.click(cancelButton);
      
      expect(mockClearMessages).not.toHaveBeenCalled();
    });

    it('closes menu after successful clear', async () => {
      const user = userEvent.setup();
      
      render(<ChatActionsMenu />);
      
      const button = screen.getByRole('button', { name: /chat options/i });
      await user.click(button);
      
      const clearButton = screen.getByRole('menuitem', { name: /clear chat/i });
      await user.click(clearButton);
      
      // Confirm the clear in the modal
      const confirmButton = screen.getByRole('button', { name: /clear chat/i });
      await user.click(confirmButton);
      
      await waitFor(() => {
        expect(screen.queryByText('Clear chat')).not.toBeInTheDocument();
      });
    });

    it('does not call clearMessages when no messages exist', async () => {
      const user = userEvent.setup();
      
      // Mock empty messages
      vi.mocked(useChatStore).mockReturnValue({
        messages: [],
        clearMessages: mockClearMessages,
      } as ReturnType<typeof useChatStore>);
      
      render(<ChatActionsMenu />);
      
      const button = screen.getByRole('button', { name: /chat options/i });
      await user.click(button);
      
      const clearButton = screen.getByRole('menuitem', { name: /clear chat/i });
      await user.click(clearButton);
      
      // Should not show modal or call clearMessages when no messages exist
      expect(mockClearMessages).not.toHaveBeenCalled();
      expect(screen.queryByText(/Are you sure you want to clear all chat messages/)).not.toBeInTheDocument();
    });
  });
}); 