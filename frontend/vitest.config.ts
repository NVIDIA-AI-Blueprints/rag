import { defineConfig, mergeConfig } from 'vitest/config';
import viteConfig from './vite.config';

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      globals: true,
      environment: 'jsdom',
      setupFiles: ['./src/test/setup.ts'],
      include: [
        'src/pages/**/*.{test,spec}.{ts,tsx}',
        'src/components/**/*.{test,spec}.{ts,tsx}',
        'src/hooks/**/*.{test,spec}.{ts,tsx}',
        'src/store/**/*.{test,spec}.{ts,tsx}'
      ],
      // Handle CSS imports from KUI components
      css: {
        modules: {
          classNameStrategy: 'non-scoped'
        }
      },
      server: {
        deps: {
          inline: ['@kui/react', '@kui/foundations']
        }
      },
      coverage: {
        reporter: ['text', 'html', 'clover', 'json', 'cobertura'],
        include: [
          'src/pages/**/*.{ts,tsx}',
          'src/components/**/*.{ts,tsx}',
          'src/hooks/**/*.{ts,tsx}',
          'src/store/**/*.{ts,tsx}'
        ],
        exclude: [
          'node_modules/',
          'src/test/',
          '**/*.d.ts',
          '**/*.config.*',
          '**/coverage/**',
          '.next/**',
          'dist/**',
          'build/**',
          'public/**',
          '**/*.test.*',
          '**/*.spec.*',
        ],
      },
    },
  })
); 