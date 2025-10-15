import { describe, it, expect } from 'vitest';
import { render, screen } from '../../../test/utils';
import { SettingsHeader } from '../SettingsHeader';

describe('SettingsHeader', () => {
  it('renders title and description', () => {
    render(<SettingsHeader />);
    
    expect(screen.getByTestId("settings-title")).toHaveTextContent("Settings");
    expect(screen.getByTestId("settings-description")).toBeInTheDocument();
  });

  it('renders settings icon', () => {
    render(<SettingsHeader />);
    expect(screen.getByTestId('settings-icon')).toBeInTheDocument();
  });
}); 