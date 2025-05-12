import React, {useRef, useEffect, useState} from 'react';
import { Box, useTheme, Button, alpha } from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DownloadIcon from '@mui/icons-material/Download';

interface SvgEditorProps {
    content: string;
}

export default function SvgEditor({content}: SvgEditorProps) {
    const theme = useTheme();
    const containerRef = useRef<HTMLDivElement>(null);
    const [isHovering, setIsHovering] = useState(false);
    
    // For clipboard and download actions
    const handleCopyToClipboard = () => {
        navigator.clipboard.writeText(content).catch(err => {
            console.error('Failed to copy SVG to clipboard:', err);
        });
    };
    
    const handleDownloadSvg = () => {
        const blob = new Blob([content], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat2svg-${Date.now()}.svg`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };
    
    useEffect(() => {
        if (containerRef.current) {
            containerRef.current.innerHTML = content;
        }
    }, [content]);

    return (
        <Box 
            sx={{ 
                height: '100%',
                position: 'relative',
                bgcolor: theme.palette.mode === 'dark' ? alpha(theme.palette.background.paper, 0.4) : '#fff',
                borderRadius: 0,
                overflow: 'hidden',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                padding: 2,
            }}
            onMouseEnter={() => setIsHovering(true)}
            onMouseLeave={() => setIsHovering(false)}
        >
            <div 
                ref={containerRef} 
                style={{ 
                    maxWidth: '100%', 
                    maxHeight: '100%',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                }}
            />
            
            {isHovering && (
                <Box sx={{
                    position: 'absolute',
                    top: 8,
                    right: 8,
                    display: 'flex',
                    gap: 1,
                    zIndex: 10,
                }}>
                    <Button
                        variant="contained"
                        size="small"
                        startIcon={<ContentCopyIcon fontSize="small" />}
                        onClick={handleCopyToClipboard}
                        sx={{
                            bgcolor: alpha(theme.palette.background.paper, 0.7),
                            color: theme.palette.text.primary,
                            backdropFilter: 'blur(4px)',
                            textTransform: 'none',
                            fontWeight: 500,
                            fontSize: '0.75rem',
                            boxShadow: 'none',
                            '&:hover': {
                                bgcolor: alpha(theme.palette.background.paper, 0.9),
                                boxShadow: 'none',
                            }
                        }}
                    >
                        Copy
                    </Button>
                    <Button
                        variant="contained"
                        size="small"
                        startIcon={<DownloadIcon fontSize="small" />}
                        onClick={handleDownloadSvg}
                        sx={{
                            bgcolor: alpha(theme.palette.background.paper, 0.7),
                            color: theme.palette.text.primary,
                            backdropFilter: 'blur(4px)',
                            textTransform: 'none',
                            fontWeight: 500,
                            fontSize: '0.75rem',
                            boxShadow: 'none',
                            '&:hover': {
                                bgcolor: alpha(theme.palette.background.paper, 0.9),
                                boxShadow: 'none',
                            }
                        }}
                    >
                        Download
                    </Button>
                </Box>
            )}
        </Box>
    );
}