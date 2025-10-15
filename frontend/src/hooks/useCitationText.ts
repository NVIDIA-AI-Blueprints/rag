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

export function useCitationText() {
  const toMarkdown = (raw: string): string => {
    if (!raw) return "";

    const lines = raw.split("\n");
    const result: string[] = [];
    let buffer = "";

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line === "") {
        if (buffer) result.push(buffer.trim());
        buffer = "";
        result.push("");
        continue;
      }
      if (/^[•\-–●▸]/.test(line)) {
        if (buffer) result.push(buffer.trim());
        buffer = "";
        result.push(`- ${line.slice(1).trim()}`);
        continue;
      }
      if (/^\d+[.)]\s*/.test(line)) {
        if (buffer) result.push(buffer.trim());
        buffer = "";
        result.push(`1. ${line.replace(/^\d+[.)]\s*/, "")}`);
        continue;
      }
      if (buffer && !buffer.trim().endsWith(".") && !buffer.trim().endsWith(":")) {
        buffer += " " + line;
      } else {
        if (buffer) result.push(buffer.trim());
        buffer = line;
      }
    }

    if (buffer) result.push(buffer.trim());
    return result.join("\n");
  };

  const getAbridgedText = (raw: string, maxLines = 2): string => {
    return raw
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean)
      .slice(0, maxLines)
      .join(" ")
      .slice(0, 300);
  };

  return { toMarkdown, getAbridgedText };
}
