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

import { useMarkdownRenderer } from "../../hooks/useMarkdownRenderer";
import { useCitationText } from "../../hooks/useCitationText";

interface CitationTextContentProps {
  text: string;
}

export const CitationTextContent = ({ text }: CitationTextContentProps) => {
  const { renderMarkdown } = useMarkdownRenderer();
  const { toMarkdown } = useCitationText();

  return (
    <div className="prose prose-invert max-w-none">
      <div
        className="text-gray-300 leading-relaxed text-xs"
        dangerouslySetInnerHTML={{
          __html: renderMarkdown(toMarkdown(text)),
        }}
      />
    </div>
  );
}; 