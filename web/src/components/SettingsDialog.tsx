import React, { useState, useEffect } from 'react';
import { 
  Dialog, 
  DialogContent, 
  DialogTitle, 
  IconButton, 
  Tab, 
  Tabs, 
  Box, 
  TextField, 
  Button, 
  Typography, 
  CircularProgress,
  FormControlLabel,
  Checkbox,
  Select,
  MenuItem,
  InputLabel,
  FormControl,
  Alert,
  Snackbar,
  useTheme,
  alpha,
  Paper,
  Divider,
  Fade,
  Zoom
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

interface ModelsByProvider {
  [key: string]: string[];
}

const TabPanel = (props: TabPanelProps) => {
  const { children, value, index, ...other } = props;
  const theme = useTheme();

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ 
          p: 3,
          height: '100%',
          overflowY: 'auto'
        }}>
          {children}
        </Box>
      )}
    </div>
  );
};

const a11yProps = (index: number) => {
  return {
    id: `settings-tab-${index}`,
    'aria-controls': `settings-tabpanel-${index}`,
  };
};

interface Provider {
  provider: string;
  model: string;
  api_base?: string;
  has_api_key: boolean;
}

interface Settings {
  default_provider: string;
  providers: {
    [key: string]: Provider;
  };
}

interface ProviderParams {
  provider: string;
  api_key?: string;
  api_base?: string;
  model?: string;
  set_default?: boolean;
}

interface SettingsDialogProps {
  open: boolean;
  onClose: () => void;
  onSettingsChange?: (provider: string) => void;
}

export default function SettingsDialog({ open, onClose, onSettingsChange }: SettingsDialogProps) {
  const [tabValue, setTabValue] = useState(0);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [loading, setLoading] = useState(false);
  const [testingConnection, setTestingConnection] = useState(false);
  const [formData, setFormData] = useState<Record<string, ProviderParams>>({});
  const [models, setModels] = useState<ModelsByProvider>({});
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info' | 'warning';
  }>({
    open: false,
    message: '',
    severity: 'info',
  });
  
  const theme = useTheme();

  useEffect(() => {
    if (open) {
      fetchSettings();
      fetchModels();
    }
  }, [open]);

  const fetchSettings = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/settings');
      if (!response.ok) {
        throw new Error('Failed to fetch settings');
      }
      const data = await response.json();
      setSettings(data);
      
      // Initialize form data
      const initialFormData: Record<string, ProviderParams> = {};
      Object.keys(data.providers).forEach(key => {
        const provider = data.providers[key];
        initialFormData[key] = {
          provider: key,
          api_base: provider.api_base || '',
          model: provider.model || '',
          set_default: data.default_provider === key,
        };
      });
      setFormData(initialFormData);
    } catch (error) {
      console.error('Error fetching settings:', error);
      showSnackbar('Failed to load settings', 'error');
    } finally {
      setLoading(false);
    }
  };

  const fetchModels = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/models');
      if (!response.ok) {
        throw new Error('Failed to fetch models');
      }
      const data = await response.json();
      setModels(data);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleFormChange = (provider: string, field: string, value: string | boolean) => {
    setFormData(prev => ({
      ...prev,
      [provider]: {
        ...prev[provider],
        [field]: value
      }
    }));
  };

  const handleSubmit = async (provider: string) => {
    const params = formData[provider];
    
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/settings/provider', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(params)
      });
      
      if (!response.ok) {
        throw new Error('Failed to update settings');
      }
      
      const data = await response.json();
      showSnackbar(`Settings updated for ${provider}`, 'success');
      
      // Refresh settings
      await fetchSettings();
      
      // Notify parent component if the default provider changed
      if (params.set_default && onSettingsChange) {
        onSettingsChange(provider);
      }
    } catch (error) {
      console.error('Error updating settings:', error);
      showSnackbar('Failed to update settings', 'error');
    } finally {
      setLoading(false);
    }
  };

  const testConnection = async (provider: string) => {
    setTestingConnection(true);
    try {
      const response = await fetch(`http://localhost:8000/api/test_connection/${provider}`);
      const data = await response.json();
      
      if (data.status === 'ok') {
        showSnackbar(`Connection successful! Model: ${data.model}`, 'success');
      } else {
        showSnackbar(`Connection failed: ${data.message}`, 'error');
      }
    } catch (error) {
      console.error('Error testing connection:', error);
      showSnackbar('Failed to test connection', 'error');
    } finally {
      setTestingConnection(false);
    }
  };

  const showSnackbar = (message: string, severity: 'success' | 'error' | 'info' | 'warning') => {
    setSnackbar({
      open: true,
      message,
      severity,
    });
  };

  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  const getProviderTitle = (provider: string): string => {
    switch (provider) {
      case 'openai':
        return 'OpenAI';
      case 'anthropic':
        return 'Anthropic (Claude)';
      case 'openrouter':
        return 'OpenRouter';
      case 'local':
        return 'Local LLM (OpenAI Compatible)';
      default:
        return provider;
    }
  };

  const getProviderDescription = (provider: string): string => {
    switch (provider) {
      case 'openai':
        return 'Configure OpenAI API access for models like GPT-4o and GPT-3.5';
      case 'anthropic':
        return 'Connect to Anthropic\'s Claude models for high-quality responses';
      case 'openrouter':
        return 'Access multiple models from different providers through a single API';
      case 'local':
        return 'Connect to locally-hosted LLMs with OpenAI-compatible APIs';
      default:
        return '';
    }
  };

  if (loading && !settings) {
    return (
      <Dialog 
        open={open} 
        onClose={onClose} 
        maxWidth="md" 
        fullWidth
        PaperProps={{
          sx: {
            borderRadius: 2,
            bgcolor: theme.palette.background.paper,
          }
        }}
      >
        <DialogTitle sx={{ px: 3, py: 2.5 }}>
          Settings
          <IconButton
            aria-label="close"
            onClick={onClose}
            sx={{ position: 'absolute', right: 8, top: 8 }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '300px' }}>
          <CircularProgress />
        </DialogContent>
      </Dialog>
    );
  }

  const providerNames = settings ? Object.keys(settings.providers) : [];
  
  // Add missing providers
  ['openai', 'anthropic', 'openrouter', 'local'].forEach(provider => {
    if (settings && !settings.providers[provider]) {
      providerNames.push(provider);
      if (!formData[provider]) {
        setFormData(prev => ({
          ...prev,
          [provider]: {
            provider,
            api_base: '',
            model: '',
            set_default: false,
          }
        }));
      }
    }
  });

  return (
    <Dialog 
      open={open} 
      onClose={onClose} 
      maxWidth="md" 
      fullWidth
      TransitionComponent={Fade}
      transitionDuration={{
        enter: 400,
        exit: 300,
      }}
      PaperProps={{
        sx: {
          borderRadius: 2,
          bgcolor: theme.palette.background.paper,
          backgroundImage: 'none',
          overflow: 'hidden',
        }
      }}
    >
      <DialogTitle sx={{ px: 3, py: 2.5, borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}` }}>
        <Typography variant="h6" fontWeight={600}>
          Settings
        </Typography>
        <IconButton
          aria-label="close"
          onClick={onClose}
          sx={{ position: 'absolute', right: 16, top: 16 }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent sx={{ p: 0, display: 'flex', flexDirection: 'column', height: '550px' }}>
        <Box sx={{ 
          display: 'flex', 
          borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          backgroundColor: theme.palette.mode === 'dark' ? alpha(theme.palette.background.default, 0.6) : alpha(theme.palette.background.default, 0.8),
        }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange} 
            aria-label="settings tabs"
            sx={{
              '.MuiTabs-indicator': {
                backgroundColor: theme.palette.primary.main,
                height: 3,
              },
              '.MuiTab-root': {
                textTransform: 'none',
                minWidth: 0,
                fontSize: '0.9rem',
                fontWeight: 600,
                px: 3,
                pt: 2,
                '&.Mui-selected': {
                  color: theme.palette.primary.main,
                }
              }
            }}
          >
            {providerNames.map((provider, index) => (
              <Tab 
                key={provider} 
                label={getProviderTitle(provider)} 
                {...a11yProps(index)} 
                sx={{
                  borderBottom: settings?.providers[provider] && settings.default_provider === provider 
                    ? `3px solid ${theme.palette.primary.main}` 
                    : 'none'
                }}
              />
            ))}
          </Tabs>
        </Box>
        
        {providerNames.map((provider, index) => (
          <TabPanel key={provider} value={tabValue} index={index}>
            <Paper 
              elevation={0} 
              sx={{ 
                p: 3, 
                mb: 3, 
                borderRadius: 2,
                backgroundColor: theme.palette.mode === 'dark' ? alpha(theme.palette.background.default, 0.4) : alpha(theme.palette.background.default, 0.5),
                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`
              }}
            >
              <Typography variant="h6" fontWeight={600} gutterBottom>
                {getProviderTitle(provider)}
                {settings?.providers[provider] && settings.default_provider === provider && (
                  <Typography 
                    component="span" 
                    sx={{ 
                      ml: 2, 
                      fontSize: '0.75rem', 
                      bgcolor: theme.palette.primary.main,
                      color: theme.palette.primary.contrastText,
                      p: '3px 8px',
                      borderRadius: 1,
                      fontWeight: 500,
                      verticalAlign: 'middle'
                    }}
                  >
                    Default
                  </Typography>
                )}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                {getProviderDescription(provider)}
              </Typography>
              <Divider sx={{ my: 2 }} />
            </Paper>
            
            <Box component="form" sx={{ mt: 2 }}>
              <TextField
                fullWidth
                margin="normal"
                label="API Key"
                type="password"
                variant="outlined"
                value={formData[provider]?.api_key || ''}
                onChange={(e) => handleFormChange(provider, 'api_key', e.target.value)}
                placeholder={settings?.providers[provider]?.has_api_key ? '••••••••••••••••••••••••••••••' : 'Enter API Key'}
                helperText={
                  provider === 'openai' ? 'Enter your OpenAI API key' :
                  provider === 'anthropic' ? 'Enter your Anthropic API key' :
                  provider === 'openrouter' ? 'Enter your OpenRouter API key' :
                  provider === 'local' ? 'API key for your local LLM (may not be required)' : ''
                }
                sx={{
                  '& .MuiOutlinedInput-root': {
                    borderRadius: 1.5,
                  }
                }}
              />
              
              {(provider === 'openai' || provider === 'local' || provider === 'openrouter') && (
                <TextField
                  fullWidth
                  margin="normal"
                  label="API Base URL"
                  variant="outlined"
                  value={formData[provider]?.api_base || ''}
                  onChange={(e) => handleFormChange(provider, 'api_base', e.target.value)}
                  placeholder={
                    provider === 'openai' ? 'https://api.openai.com/v1' :
                    provider === 'openrouter' ? 'https://openrouter.ai/api/v1' :
                    provider === 'local' ? 'http://localhost:1234/v1' : ''
                  }
                  helperText={
                    provider === 'openai' ? 'Optional: custom OpenAI-compatible endpoint' :
                    provider === 'openrouter' ? 'Optional: custom OpenRouter endpoint' :
                    provider === 'local' ? 'URL for your local LLM server with OpenAI-compatible API' : ''
                  }
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1.5,
                    }
                  }}
                />
              )}
              
              <FormControl 
                fullWidth 
                margin="normal"
                sx={{
                  '& .MuiOutlinedInput-root': {
                    borderRadius: 1.5,
                  }
                }}
              >
                <InputLabel id={`${provider}-model-label`}>Model</InputLabel>
                <Select
                  labelId={`${provider}-model-label`}
                  value={formData[provider]?.model || ''}
                  label="Model"
                  onChange={(e) => handleFormChange(provider, 'model', e.target.value)}
                >
                  {models[provider]?.map((model: string) => (
                    <MenuItem key={model} value={model}>{model}</MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <FormControlLabel
                control={
                  <Checkbox
                    checked={formData[provider]?.set_default || false}
                    onChange={(e) => handleFormChange(provider, 'set_default', e.target.checked)}
                  />
                }
                label="Set as default provider"
                sx={{ mt: 1 }}
              />
              
              <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button 
                  variant="contained" 
                  onClick={() => handleSubmit(provider)}
                  disabled={loading}
                  startIcon={loading ? <CircularProgress size={20} /> : null}
                  sx={{ 
                    px: 3,
                    py: 1,
                    background: 'linear-gradient(90deg, #7c3aed, #0ea5e9)',
                    '&:hover': {
                      background: 'linear-gradient(90deg, #6d28d9, #0284c7)'
                    },
                    textTransform: 'none',
                    fontWeight: 600,
                    boxShadow: 'none'
                  }}
                >
                  Save Settings
                </Button>
                
                <Button
                  variant="outlined"
                  onClick={() => testConnection(provider)}
                  disabled={testingConnection || !settings?.providers[provider]?.has_api_key}
                  startIcon={testingConnection ? <CircularProgress size={20} /> : null}
                  sx={{ 
                    px: 3,
                    textTransform: 'none',
                    fontWeight: 500
                  }}
                >
                  Test Connection
                </Button>
              </Box>
              
              {provider === 'local' && (
                <Alert 
                  severity="info" 
                  sx={{ 
                    mt: 3,
                    borderRadius: 1.5,
                    backgroundColor: theme.palette.mode === 'dark' ? alpha(theme.palette.info.main, 0.1) : alpha(theme.palette.info.light, 0.1),
                    border: '1px solid',
                    borderColor: theme.palette.mode === 'dark' ? alpha(theme.palette.info.main, 0.2) : alpha(theme.palette.info.light, 0.3),
                    '& .MuiAlert-icon': {
                      color: theme.palette.info.main
                    }
                  }}
                >
                  For local LLMs, make sure your local server implements the OpenAI-compatible API interface.
                  Examples include LM Studio, Ollama, and vLLM.
                </Alert>
              )}
              
              {provider === 'openrouter' && (
                <Alert 
                  severity="info" 
                  sx={{ 
                    mt: 3,
                    borderRadius: 1.5,
                    backgroundColor: theme.palette.mode === 'dark' ? alpha(theme.palette.info.main, 0.1) : alpha(theme.palette.info.light, 0.1),
                    border: '1px solid',
                    borderColor: theme.palette.mode === 'dark' ? alpha(theme.palette.info.main, 0.2) : alpha(theme.palette.info.light, 0.3),
                    '& .MuiAlert-icon': {
                      color: theme.palette.info.main
                    }
                  }}
                >
                  OpenRouter provides access to multiple LLM providers through a single API.
                  Get your API key at <a href="https://openrouter.ai" target="_blank" rel="noopener noreferrer" style={{color: theme.palette.primary.main}}>openrouter.ai</a>
                </Alert>
              )}
            </Box>
          </TabPanel>
        ))}
      </DialogContent>
      
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        TransitionComponent={Zoom}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: '100%', borderRadius: 1.5 }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Dialog>
  );
} 