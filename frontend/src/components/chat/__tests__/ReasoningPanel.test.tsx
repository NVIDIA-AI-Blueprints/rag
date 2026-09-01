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

import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { ReasoningPanel } from '../ReasoningPanel';
import type { ReasoningStep } from '../../../types/chat';

const buildStep = (overrides: Partial<ReasoningStep> = {}): ReasoningStep => ({
  stage: 'plan',
  reasoning: '',
  output: '',
  status: 'done',
  ...overrides,
});

describe('ReasoningPanel', () => {
  it('renders nothing when there are no steps', () => {
    const { container } = render(<ReasoningPanel steps={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('shows "Thought for N steps" header when not streaming', () => {
    render(
      <ReasoningPanel
        steps={[
          buildStep({ stage: 'plan', label: 'Planning…', summary: 'Done.' }),
          buildStep({ stage: 'execute', label: 'Running tasks…' }),
        ]}
      />
    );
    expect(screen.getByTestId('reasoning-panel-toggle')).toHaveTextContent(
      'Thought for 2 steps'
    );
  });

  it('shows live "Thinking" header while streaming', () => {
    render(
      <ReasoningPanel
        streaming
        steps={[buildStep({ stage: 'plan', status: 'running' })]}
      />
    );
    expect(screen.getByTestId('reasoning-panel-toggle')).toHaveTextContent(
      'Thinking (1 step)'
    );
  });

  it('formats the stage identifier into a human-readable label without code change', () => {
    render(
      <ReasoningPanel
        streaming
        steps={[buildStep({ stage: 'verify_execute', status: 'running' })]}
      />
    );
    expect(screen.getByTestId('reasoning-step-verify_execute')).toHaveTextContent(
      'Verify execute'
    );
  });

  it('auto-expands while streaming and shows step content', () => {
    const steps: ReasoningStep[] = [
      buildStep({
        stage: 'plan',
        label: 'Planning…',
        reasoning: 'Step 1: scope.',
        status: 'running',
      }),
    ];
    render(<ReasoningPanel streaming steps={steps} />);
    expect(screen.getByTestId('reasoning-panel')).toHaveAttribute(
      'data-state',
      'open'
    );
    expect(screen.getByTestId('reasoning-step-plan')).toHaveTextContent(
      'Step 1: scope.'
    );
  });

  it('starts collapsed when not streaming and toggles on click', () => {
    render(
      <ReasoningPanel
        steps={[buildStep({ stage: 'plan', label: 'Planning…' })]}
      />
    );
    expect(screen.getByTestId('reasoning-panel')).toHaveAttribute(
      'data-state',
      'closed'
    );
    fireEvent.click(screen.getByTestId('reasoning-panel-toggle'));
    expect(screen.getByTestId('reasoning-panel')).toHaveAttribute(
      'data-state',
      'open'
    );
  });

  it('exposes step status via data-status for downstream styling', () => {
    render(
      <ReasoningPanel
        streaming
        steps={[
          buildStep({ stage: 'plan', status: 'running' }),
          buildStep({ stage: 'execute', status: 'error' }),
          buildStep({ stage: 'synthesize', status: 'done' }),
        ]}
      />
    );
    expect(screen.getByTestId('reasoning-step-plan')).toHaveAttribute(
      'data-status',
      'running'
    );
    expect(screen.getByTestId('reasoning-step-execute')).toHaveAttribute(
      'data-status',
      'error'
    );
    expect(screen.getByTestId('reasoning-step-synthesize')).toHaveAttribute(
      'data-status',
      'done'
    );
  });

  it('renders both reasoning and output blocks when both are present', () => {
    render(
      <ReasoningPanel
        streaming
        steps={[
          buildStep({
            stage: 'execute',
            reasoning: 'thinking text',
            output: '{"task": 1}',
            status: 'running',
          }),
        ]}
      />
    );
    const stepEl = screen.getByTestId('reasoning-step-execute');
    expect(stepEl).toHaveTextContent('thinking text');
    expect(stepEl).toHaveTextContent('{"task": 1}');
  });

  it('toggle is keyboard accessible (Enter / Space)', () => {
    render(
      <ReasoningPanel
        steps={[buildStep({ stage: 'plan', label: 'Planning…' })]}
      />
    );
    const toggle = screen.getByTestId('reasoning-panel-toggle');
    expect(screen.getByTestId('reasoning-panel')).toHaveAttribute(
      'data-state',
      'closed'
    );
    fireEvent.keyDown(toggle, { key: 'Enter' });
    expect(screen.getByTestId('reasoning-panel')).toHaveAttribute(
      'data-state',
      'open'
    );
    fireEvent.keyDown(toggle, { key: ' ' });
    expect(screen.getByTestId('reasoning-panel')).toHaveAttribute(
      'data-state',
      'closed'
    );
  });
});
