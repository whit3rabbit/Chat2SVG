import React, {useState, useRef, useEffect} from 'react';
import {Box, TextField, Button, CircularProgress, Alert, Typography, IconButton, useMediaQuery, createTheme, ThemeProvider, CssBaseline, Paper, Divider} from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import ProgressStepper from './components/ProgressStepper';
import ProgressBar from './components/ProgressBar';
import SvgEditor from './components/SvgEditor';
import SettingsDialog from './components/SettingsDialog';
import LogoAnimation from './components/LogoAnimation';
import {WebSocketMessage} from './type/type';

export default function App() {
    const [prompt, setPrompt] = useState('');
    const [messages, setMessages] = useState<WebSocketMessage[]>([]);
    const [svgContent, setSvgContent] = useState('');
    const [svgContent2, setSvgContent2] = useState('');
    const [svgContent3, setSvgContent3] = useState('');
    const [error, setError] = useState('');
    const [currentStage, setCurrentStage] = useState(0);
    const [failedStage, setFailedStage] = useState<number | null>(null);
    const [provider, setProvider] = useState<string | null>(null);
    const [settingsOpen, setSettingsOpen] = useState(false);
    const [darkMode, setDarkMode] = useState<boolean>(() => {
        // Check for user preference in localStorage or use system preference
        const savedMode = localStorage.getItem('darkMode');
        if (savedMode !== null) {
            return savedMode === 'true';
        }
        return window.matchMedia('(prefers-color-scheme: dark)').matches;
    });
    const ws = useRef<WebSocket | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const theme = createTheme({
        palette: {
            mode: darkMode ? 'dark' : 'light',
            primary: {
                main: '#7c3aed', // Purple
            },
            secondary: {
                main: '#0ea5e9', // Sky blue
            },
            background: {
                default: darkMode ? '#1e1e2e' : '#f8fafc',
                paper: darkMode ? '#27273a' : '#ffffff',
            },
            text: {
                primary: darkMode ? '#f1f5f9' : '#334155',
                secondary: darkMode ? '#cbd5e1' : '#64748b',
            },
        },
        typography: {
            fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
            h3: {
                fontWeight: 700,
            },
        },
        shape: {
            borderRadius: 12,
        },
        components: {
            MuiButton: {
                styleOverrides: {
                    root: {
                        textTransform: 'none',
                        fontWeight: 600,
                        boxShadow: 'none',
                        '&:hover': {
                            boxShadow: 'none',
                        },
                    },
                },
            },
            MuiTextField: {
                styleOverrides: {
                    root: {
                        '& .MuiOutlinedInput-root': {
                            borderRadius: 8,
                        },
                    },
                },
            },
            MuiPaper: {
                styleOverrides: {
                    root: {
                        backgroundImage: 'none',
                    },
                },
            },
        },
    });

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({behavior: "smooth"});
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);
    
    useEffect(() => {
        localStorage.setItem('darkMode', darkMode.toString());
    }, [darkMode]);
    
    useEffect(() => {
        // Fetch the default provider on initial load
        const fetchSettings = async () => {
            try {
                const response = await fetch('http://localhost:8000/api/settings');
                if (response.ok) {
                    const data = await response.json();
                    setProvider(data.default_provider);
                }
            } catch (error) {
                console.error('Failed to fetch settings:', error);
            }
        };
        
        fetchSettings();
    }, []);

    const startGeneration = (retryStage?: number) => {
        setError('');
        setMessages([]);
        if (retryStage === 0 || retryStage === undefined) {
            setCurrentStage(1);
            setFailedStage(null);
            setSvgContent('');
            setSvgContent2('');
            setSvgContent3('');
        } else {
            setCurrentStage(prev => Math.max(prev, retryStage));
        }
        setFailedStage(null);

        const websocket = new WebSocket('ws://localhost:8000/ws/generate');

        websocket.onopen = () => {
            const params: any = {
                prompt,
                startFrom: retryStage || 1
            };
            
            // Add provider if selected
            if (provider) {
                params.provider = provider;
            }
            
            websocket.send(JSON.stringify(params));
        };

        websocket.onmessage = async (event) => {
            const data: WebSocketMessage = JSON.parse(event.data);

            const newMessage = {
                ...data,
                timestamp: Date.now()
            };

            setMessages(prev => [...prev, newMessage]);

            if ((data.status === 'svg_ready' || data.status === 'svg_ready2' || data.status === 'svg_ready3') && data.svg_file) {
                try {
                    const response = await fetch(`http://localhost:8000/api/svg/${data.svg_file}`);
                    if (!response.ok) throw new Error('Failed to fetch SVG');
                    const svgData = await response.json();
                    if (data.status === 'svg_ready') {
                        setSvgContent(svgData.content);
                    } else if (data.status === 'svg_ready2') {
                        setSvgContent2(svgData.content);
                    } else {
                        setSvgContent3(svgData.content);
                        ws.current = null;
                    }
                } catch (err: any) {
                    setError(`Get SVG Error: ${err.message}`);
                }
            }

            if (data.status === 'completed' && data.stage) {
                setCurrentStage(prev => Math.max(prev, (data.stage || 0) + 1));
            }

            if (data.status === 'error') {
                setError(data.output || 'Unknown error');
                setFailedStage(data.stage || null);
                websocket.close();
                ws.current = null;
            }
        };

        websocket.onerror = () => {
            setError('WebSocket Client Error');
            websocket.close();
        };

        websocket.onclose = () => {
            ws.current = null;
        };

        ws.current = websocket;
    };

    const handleRetry = (stage: number) => {
        if (ws.current) {
            ws.current.close();
            ws.current = null;
        }
        startGeneration(stage);
    };
    
    const handleSettingsOpen = () => {
        setSettingsOpen(true);
    };
    
    const handleSettingsClose = () => {
        setSettingsOpen(false);
    };
    
    const handleProviderChange = (newProvider: string) => {
        setProvider(newProvider);
    };

    const toggleDarkMode = () => {
        setDarkMode(prev => !prev);
    };

    useEffect(() => {
        return () => {
            if (ws.current) {
                ws.current.close();
            }
        };
    }, []);

    const formatMessage = (msg: WebSocketMessage) => {
        const timestamp = new Date(msg.timestamp || 0).toLocaleTimeString();
        const stageInfo = msg.stage ? `[Stage ${msg.stage}] ` : '';

        if (msg.status === 'svg_ready') {
            return `🎉 ${stageInfo} gen SVG file: ${msg.svg_file}`;
        }

        return `[${timestamp}] ${stageInfo}${msg.output}`;
    };
    
    const getMessageColor = (status?: string): string => {
        if (darkMode) {
            switch (status) {
                case 'error':
                    return 'rgba(239, 68, 68, 0.2)';  // Red with transparency
                case 'completed':
                    return 'rgba(34, 197, 94, 0.2)';  // Green with transparency
                case 'svg_ready':
                case 'svg_ready2':
                case 'svg_ready3':
                    return 'rgba(59, 130, 246, 0.2)'; // Blue with transparency
                default:
                    return 'transparent';
            }
        } else {
            switch (status) {
                case 'error':
                    return 'rgba(254, 226, 226, 0.6)';
                case 'completed':
                    return 'rgba(220, 252, 231, 0.6)';
                case 'svg_ready':
                case 'svg_ready2':
                case 'svg_ready3':
                    return 'rgba(219, 234, 254, 0.6)';
                default:
                    return 'transparent';
            }
        }
    };

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <Box sx={{
                display: 'flex',
                flexDirection: 'column',
                minHeight: '100vh',
                padding: 3,
                gap: 3,
                maxWidth: {xs: '100%', md: 1200},
                margin: '0 auto',
                paddingX: {xs: 2, sm: 4, md: 5},
                width: '100%'
            }}>
                <Paper 
                    elevation={0} 
                    sx={{
                        mb: 2,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        py: 2.5,
                        px: 3,
                        borderRadius: 2,
                        background: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.03)' : 'rgba(0, 0, 0, 0.01)',
                        backdropFilter: 'blur(8px)',
                        borderBottom: '1px solid',
                        borderColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)'
                    }}
                >
                    <LogoAnimation size="medium" />
                    
                    <Box sx={{ display: 'flex', gap: 1 }}>
                        <IconButton 
                            onClick={toggleDarkMode}
                            color="inherit"
                            aria-label="toggle dark mode"
                            sx={{ p: 1 }}
                        >
                            {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
                        </IconButton>
                        
                        <IconButton 
                            onClick={handleSettingsOpen}
                            color="primary"
                            aria-label="settings"
                            sx={{ p: 1 }}
                        >
                            <SettingsIcon />
                        </IconButton>
                    </Box>
                </Paper>

                <Box sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 3,
                    flex: 1
                }}>
                    <Paper 
                        elevation={0} 
                        sx={{ 
                            p: 3, 
                            borderRadius: 2,
                            border: '1px solid',
                            borderColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
                        }}
                    >
                        <Box sx={{
                            display: 'flex',
                            gap: 2,
                            alignItems: 'flex-start'
                        }}>
                            <TextField
                                fullWidth
                                label="Input Prompt"
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                sx={{flex: 2}}
                                placeholder="Describe the SVG you want to generate..."
                                variant="outlined"
                            />
                            <Button
                                variant="contained"
                                onClick={() => startGeneration()}
                                disabled={!!ws.current || !prompt.trim()}
                                startIcon={ws.current && <CircularProgress size={20} color="inherit" />}
                                sx={{ 
                                    height: 56, 
                                    px: 3,
                                    background: 'linear-gradient(90deg, #7c3aed, #0ea5e9)',
                                    '&:hover': {
                                        background: 'linear-gradient(90deg, #6d28d9, #0284c7)'
                                    },
                                    '&:disabled': {
                                        background: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
                                        color: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.3)' : 'rgba(0, 0, 0, 0.3)'
                                    }
                                }}
                            >
                                {ws.current ? 'Generating...' : 'Start generating'}
                            </Button>
                        </Box>

                        <Box sx={{ mt: 4 }}>
                            <ProgressStepper
                                currentStage={currentStage}
                                failedStage={failedStage}
                                onRetry={handleRetry}
                            />
                            
                            {ws.current && currentStage > 0 && !failedStage && (
                                <ProgressBar 
                                    isActive={!!ws.current} 
                                    stage={currentStage} 
                                    messages={messages} 
                                />
                            )}
                        </Box>
                    </Paper>

                    {(svgContent || svgContent2 || svgContent3) && (
                        <Paper 
                            elevation={0} 
                            sx={{ 
                                p: 3, 
                                borderRadius: 2,
                                border: '1px solid',
                                borderColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
                            }}
                        >
                            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                                Generated SVGs
                            </Typography>
                            <Divider sx={{ mb: 3 }} />
                            <Box sx={{
                                display: 'flex',
                                gap: 3,
                                overflowX: 'auto',
                                paddingY: 2,
                                flexWrap: 'nowrap',
                                '& > *': {
                                    flex: '0 0 380px',
                                    minWidth: 380,
                                    maxWidth: '100%',
                                    height: 420,
                                    scrollSnapAlign: 'start'
                                },
                                scrollSnapType: 'x mandatory'
                            }}>
                                {svgContent && (
                                    <Paper 
                                        elevation={0} 
                                        sx={{ 
                                            border: '1px solid',
                                            borderColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
                                            borderRadius: 2,
                                            overflow: 'hidden'
                                        }}
                                    >
                                        <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)' }}>
                                            <Typography variant="subtitle2" fontWeight={600}>Stage 1</Typography>
                                        </Box>
                                        <SvgEditor content={svgContent} />
                                    </Paper>
                                )}
                                {svgContent2 && (
                                    <Paper 
                                        elevation={0} 
                                        sx={{ 
                                            border: '1px solid',
                                            borderColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
                                            borderRadius: 2,
                                            overflow: 'hidden'
                                        }}
                                    >
                                        <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)' }}>
                                            <Typography variant="subtitle2" fontWeight={600}>Stage 2</Typography>
                                        </Box>
                                        <SvgEditor content={svgContent2} />
                                    </Paper>
                                )}
                                {svgContent3 && (
                                    <Paper 
                                        elevation={0} 
                                        sx={{ 
                                            border: '1px solid',
                                            borderColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
                                            borderRadius: 2,
                                            overflow: 'hidden'
                                        }}
                                    >
                                        <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)' }}>
                                            <Typography variant="subtitle2" fontWeight={600}>Stage 3</Typography>
                                        </Box>
                                        <SvgEditor content={svgContent3} />
                                    </Paper>
                                )}
                            </Box>
                        </Paper>
                    )}

                    <Paper 
                        elevation={0} 
                        sx={{ 
                            borderRadius: 2,
                            border: '1px solid',
                            borderColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
                            overflow: 'hidden',
                            height: messages.length > 0 ? 300 : 'auto'
                        }}
                    >
                        <Box sx={{
                            p: 2,
                            bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.03)' : 'rgba(0, 0, 0, 0.01)',
                            borderBottom: '1px solid',
                            borderColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center'
                        }}>
                            <Typography variant="subtitle1" fontWeight={600}>Log</Typography>
                            <Typography variant="caption" color="text.secondary">
                                {new Date().toLocaleDateString()}
                                {provider && ` | Provider: ${provider}`}
                            </Typography>
                        </Box>
                        <Box sx={{
                            height: messages.length > 0 ? 250 : 100,
                            overflowY: 'auto',
                            p: 2,
                            bgcolor: theme.palette.mode === 'dark' ? '#1a1b26' : '#f8fafc',
                            color: theme.palette.mode === 'dark' ? '#d4d4d4' : '#334155',
                            fontFamily: '"Menlo", "Monaco", "Courier New", monospace',
                            fontSize: 14
                        }}>
                            {messages.length > 0 ? (
                                messages.map((msg, index) => (
                                    <div key={index} style={{
                                        marginBottom: 4,
                                        padding: 8,
                                        borderRadius: 8,
                                        backgroundColor: getMessageColor(msg.status)
                                    }}>
                                        {formatMessage(msg)}
                                    </div>
                                ))
                            ) : (
                                <Typography sx={{ color: theme.palette.text.secondary, py: 2, textAlign: 'center' }}>
                                    Logs will appear here when you start the generation process.
                                </Typography>
                            )}
                            <div ref={messagesEndRef}/>
                        </Box>
                    </Paper>
                </Box>
            
                <SettingsDialog 
                    open={settingsOpen} 
                    onClose={handleSettingsClose} 
                    onSettingsChange={handleProviderChange}
                />
            </Box>
        </ThemeProvider>
    );
}