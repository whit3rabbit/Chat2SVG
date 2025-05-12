import React, { useState, useEffect } from 'react';
import { Box, LinearProgress, Typography, useTheme, alpha } from '@mui/material';

interface ProgressBarProps {
  isActive: boolean;
  stage: number;
  messages: any[];
}

export default function ProgressBar({ isActive, stage, messages }: ProgressBarProps) {
  const theme = useTheme();
  const [progress, setProgress] = useState(0);
  const [displayText, setDisplayText] = useState('');
  
  // Calculate progress based on stage and messages
  useEffect(() => {
    if (!isActive) {
      setProgress(0);
      setDisplayText('');
      return;
    }
    
    // Implement a pulsing effect when active
    let interval: NodeJS.Timeout;
    
    if (stage === 1) {
      // Counting messages roughly correlates to progress in stage 1
      const relevantMessages = messages.filter(m => m.stage === stage);
      const estimatedProgress = Math.min(relevantMessages.length * 2, 95);
      setProgress(estimatedProgress);
      
      interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) return 0;
          return prev + 1;
        });
      }, 800);
      
      // Find a message about what's happening
      const statusMessage = relevantMessages.slice(-1)[0]?.output || 'Processing...';
      setDisplayText(statusMessage.length > 50 ? `${statusMessage.substring(0, 50)}...` : statusMessage);
    } else if (stage === 2) {
      setProgress(stage === 2 ? 50 : 0);
      setDisplayText('Enhancing SVG details...');
      
      interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) return 40;
          return prev + 2;
        });
      }, 1000);
    } else if (stage === 3) {
      setProgress(75);
      setDisplayText('Optimizing SVG shapes...');
      
      interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) return 70;
          return prev + 3;
        });
      }, 1200);
    }
    
    return () => clearInterval(interval);
  }, [isActive, stage, messages]);
  
  if (!isActive) return null;
  
  return (
    <Box 
      sx={{ 
        position: 'relative', 
        mt: 1, 
        mb: 1, 
        p: 2,
        borderRadius: 2,
        backgroundColor: theme.palette.mode === 'dark' 
          ? alpha(theme.palette.primary.main, 0.1)
          : alpha(theme.palette.primary.light, 0.1),
        border: '1px solid',
        borderColor: theme.palette.mode === 'dark'
          ? alpha(theme.palette.primary.main, 0.2)
          : alpha(theme.palette.primary.light, 0.3),
      }}
    >
      <Typography 
        variant="body2" 
        fontWeight={500} 
        gutterBottom
        sx={{ 
          color: theme.palette.primary.main,
          display: 'flex',
          justifyContent: 'space-between'
        }}
      >
        <span>
          Processing Stage {stage}
        </span>
        <span>{Math.round(progress)}%</span>
      </Typography>
      
      <LinearProgress 
        variant="determinate" 
        value={progress} 
        sx={{
          height: 8,
          borderRadius: 4,
          bgcolor: theme.palette.mode === 'dark' 
            ? alpha(theme.palette.background.paper, 0.2)
            : alpha(theme.palette.background.paper, 0.5),
          '& .MuiLinearProgress-bar': {
            borderRadius: 4,
            background: 'linear-gradient(90deg, #7c3aed, #0ea5e9)',
          }
        }}
      />
      
      <Typography 
        variant="caption" 
        sx={{ 
          mt: 1,
          display: 'block',
          color: theme.palette.text.secondary,
          fontStyle: 'italic',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis'
        }}
      >
        {displayText || 'Processing...'}
      </Typography>
    </Box>
  );
} 