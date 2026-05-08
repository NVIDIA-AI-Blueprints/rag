// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import { render, screen, fireEvent, act } from '../../../test/utils';
import { AgenticModeSelector } from '../AgenticModeSelector';
import { useSettingsStore, type AgenticMode } from '../../../store/useSettingsStore';

// We want the real Zustand store but a minimal Dropdown so `onSelect`
// handlers are reachable via simple click events in jsdom.
vi.mock('@kui/react', async () => {
  const actual = await vi.importActual<typeof import('@kui/react')>('@kui/react');
  type Item = { children: React.ReactNode; onSelect?: () => void };
  return {
    ...actual,
    Dropdown: ({
      items,
      children,
      ...rest
    }: {
      items: Item[];
      children: React.ReactNode;
    } & Record<string, unknown>) => (
      <div data-testid={(rest['data-testid'] as string) ?? 'mock-dropdown'}>
        {children}
        <ul>
          {items.map((item, i) => (
            <li key={i}>
              <button
                type="button"
                data-testid={`mock-dropdown-item-${i}`}
                onClick={item.onSelect}
              >
                {item.children}
              </button>
            </li>
          ))}
        </ul>
      </div>
    ),
  };
});

describe('AgenticModeSelector', () => {
  beforeEach(() => {
    // The store now defaults to "off" (Standard) and the previously-allowed
    // "auto" value was removed — see the AgenticMode type.
    act(() => {
      useSettingsStore.setState({ agenticMode: 'off' });
    });
  });

  it('renders the current mode in the trigger label', () => {
    render(<AgenticModeSelector />);
    const trigger = screen.getByTestId('agentic-mode-trigger');
    expect(trigger).toHaveTextContent('Standard');
    expect(trigger).toHaveAttribute('data-mode', 'off');
  });

  it('exposes both modes as dropdown items in a stable order (Standard first)', () => {
    render(<AgenticModeSelector />);
    expect(screen.getByTestId('agentic-mode-option-off')).toHaveTextContent('Standard');
    expect(screen.getByTestId('agentic-mode-option-on')).toHaveTextContent('Agentic');
    expect(
      screen.queryByTestId('agentic-mode-option-auto')
    ).not.toBeInTheDocument();
  });

  it('marks the active option via data-active', () => {
    act(() => {
      useSettingsStore.setState({ agenticMode: 'on' });
    });
    render(<AgenticModeSelector />);
    expect(screen.getByTestId('agentic-mode-option-on')).toHaveAttribute(
      'data-active',
      'true'
    );
    expect(screen.getByTestId('agentic-mode-option-off')).toHaveAttribute(
      'data-active',
      'false'
    );
  });

  it.each<[string, AgenticMode]>([
    ['mock-dropdown-item-0', 'off'],
    ['mock-dropdown-item-1', 'on'],
  ])('selecting %s sets agenticMode to %s', (testId, expected) => {
    render(<AgenticModeSelector />);
    fireEvent.click(screen.getByTestId(testId));
    expect(useSettingsStore.getState().agenticMode).toBe(expected);
  });

  it('updates the trigger label after selecting a mode', () => {
    render(<AgenticModeSelector />);
    fireEvent.click(screen.getByTestId('mock-dropdown-item-1'));
    expect(screen.getByTestId('agentic-mode-trigger')).toHaveTextContent('Agentic');
    expect(screen.getByTestId('agentic-mode-trigger')).toHaveAttribute('data-mode', 'on');
  });

  it('exposes an aria-label on the trigger reflecting the current mode', () => {
    render(<AgenticModeSelector />);
    expect(screen.getByTestId('agentic-mode-trigger')).toHaveAttribute(
      'aria-label',
      'Pipeline mode: Standard'
    );
  });
});
