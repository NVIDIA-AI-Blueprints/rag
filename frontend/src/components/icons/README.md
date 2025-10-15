# Settings Icons

This module contains reusable SVG icons for the settings interface.

## Available Icons

- `RagIcon` - Document icon for RAG configuration
- `FeaturesIcon` - Toggle switches icon for feature toggles
- `ModelsIcon` - Server/computer icon for model configuration
- `EndpointsIcon` - Link icon for endpoint configuration
- `AdvancedIcon` - Gear icon for advanced settings

## Usage

### As React Components (Recommended)

```tsx
import { RagIcon, FeaturesIcon, ModelsIcon } from '@/components/icons/SettingsIcons';
// or from the index
import { RagIcon, FeaturesIcon, ModelsIcon } from '@/components/icons';

// Basic usage
<RagIcon />

// With custom size
<RagIcon className="w-8 h-8" />

// With explicit size prop
<RagIcon size={32} />
```

### As Function Components (Legacy)

For backward compatibility, the old function-style exports are still available:

```tsx
import { ICON_rag, ICON_features, ICON_models } from '@/components/icons/SettingsIcons';

<ICON_rag />
<ICON_features />
```

## Props

Each icon component accepts:

- `className?: string` - CSS classes (default: "w-5 h-5")
- `size?: number` - Explicit width/height in pixels

## Styling

Icons use `currentColor` for stroke, so they inherit the text color of their parent container. This makes them easy to theme and style consistently with your design system.

## Adding New Icons

1. Create a new icon component following the same pattern
2. Export it from `SettingsIcons.tsx`
3. Add it to this README
