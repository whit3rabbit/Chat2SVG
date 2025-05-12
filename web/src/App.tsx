import React, {useState, useRef, useEffect} from 'react';
import {Box, TextField, Button, CircularProgress, Alert, Typography, IconButton} from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import ProgressStepper from './components/ProgressStepper';
import SvgEditor from './components/SvgEditor';
import SettingsDialog from './components/SettingsDialog';
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
    const ws = useRef<WebSocket | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({behavior: "smooth"});
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);
    
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
                } catch (err) {
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

        websocket.onerror = (err) => {
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
        switch (status) {
            case 'error':
                return 'rgba(255, 0, 0, 0.1)';
            case 'completed':
                return 'rgba(0, 255, 0, 0.1)';
            case 'svg_ready':
            case 'svg_ready2':
            case 'svg_ready3':
                return 'rgba(0, 0, 255, 0.1)';
            default:
                return 'transparent';
        }
    };

    return (
        <Box sx={{
            display: 'flex',
            flexDirection: 'column',
            height: '100vh',
            padding: 2,
            gap: 3,
            maxWidth: {xs: '100%', md: 1200},
            margin: '0 auto',
            paddingX: {xs: 2, sm: 4, md: 6},
            width: '100%'
        }}>
            <Box sx={{
                mb: 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                py: 2,
                borderBottom: '2px solid #eee'
            }}>
                <Typography variant="h3" component="h1" sx={{fontWeight: 700, color: 'primary.main'}}>
                    chat2svg
                </Typography>
                
                <IconButton 
                    onClick={handleSettingsOpen}
                    color="primary"
                    aria-label="settings"
                    sx={{ p: 1 }}
                >
                    <SettingsIcon />
                </IconButton>
            </Box>

            <Box sx={{
                display: 'flex',
                flexDirection: 'column',
                gap: 3,
                flex: 1
            }}>
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
                    />
                    <Button
                        variant="contained"
                        onClick={() => startGeneration()}
                        disabled={!!ws.current}
                        startIcon={ws.current && <CircularProgress size={20}/>}
                        sx={{height: 56}}
                    >
                        {ws.current ? 'Generating...' : 'Start generating'}
                    </Button>
                </Box>

                
                    <ProgressStepper
                        currentStage={currentStage}
                        failedStage={failedStage}
                        onRetry={handleRetry}
                    />
                </Box>

                <Box sx={{
                    flex: 1,
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
                    {(svgContent || svgContent2 || svgContent3) && (
                        <>
                            {svgContent && <SvgEditor content={svgContent}/>}
                            {svgContent2 && <SvgEditor content={svgContent2}/>}
                            {svgContent3 && <SvgEditor content={svgContent3}/>}
                        </>
                    )}
                </Box>

                <Box sx={{
                    border: '1px solid #ddd',
                    borderRadius: 2,
                    height: 300
                }}>
                    <Box sx={{
                        p: 2,
                        bgcolor: '#f5f5f5',
                        borderBottom: '1px solid #ddd',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center'
                    }}>
                        <Box sx={{p: 2, bgcolor: '#f5f5f5', borderBottom: '1px solid #ddd'}}>
                            <Typography variant="h6">Log</Typography>
                        </Box>
                        <Typography variant="caption" color="text.secondary">
                            {new Date().toLocaleDateString()}
                            {provider && ` | Provider: ${provider}`}
                        </Typography>
                    </Box>
                    <Box sx={{
                        height: 500,
                        overflowY: 'auto',
                        p: 2,
                        bgcolor: '#1e1e1e',
                        color: '#d4d4d4',
                        fontFamily: 'Monaco, monospace',
                        fontSize: 14
                    }}>
                        {messages.map((msg, index) => (
                            <div key={index} style={{
                                marginBottom: 4,
                                padding: 4,
                                borderRadius: 4,
                                backgroundColor: getMessageColor(msg.status)
                            }}>
                                {formatMessage(msg)}
                            </div>
                        ))}
                        <div ref={messagesEndRef}/>
                    </Box>
                </Box>
            
            <SettingsDialog 
                open={settingsOpen} 
                onClose={handleSettingsClose} 
                onSettingsChange={handleProviderChange}
            />
        </Box>
    );
}