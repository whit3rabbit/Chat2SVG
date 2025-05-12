import React from 'react';
import { useTheme, Box, keyframes } from '@mui/material';

const gradientShift = keyframes`
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
`;

const svgMorph = keyframes`
  0% {
    stroke-dashoffset: 1000;
    opacity: 0.5;
  }
  50% {
    stroke-dashoffset: 500;
    opacity: 0.8;
  }
  100% {
    stroke-dashoffset: 0;
    opacity: 1;
  }
`;

interface LogoAnimationProps {
  size?: 'small' | 'medium' | 'large';
}

export default function LogoAnimation({ size = 'medium' }: LogoAnimationProps) {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  
  const dimensions = {
    small: { width: 24, height: 24 },
    medium: { width: 40, height: 40 },
    large: { width: 60, height: 60 },
  };
  
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
      <Box 
        sx={{
          position: 'relative',
          width: dimensions[size].width,
          height: dimensions[size].height,
          '& svg': {
            animation: `${svgMorph} 3s ease-in-out infinite`,
          },
        }}
      >
        <svg 
          width={dimensions[size].width} 
          height={dimensions[size].height} 
          viewBox="0 0 100 100" 
          fill="none" 
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* SVG chat bubble with icon */}
          <path 
            d="M85 40C85 55.9 70.5 69 50 69C43.1488 69 36.7233 67.4553 31.2485 64.7301L15 70L20.4086 57.3161C16.9109 52.1924 15 46.2706 15 40C15 24.1 29.5 11 50 11C70.5 11 85 24.1 85 40Z" 
            stroke="url(#paint0_linear)" 
            strokeWidth="4"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeDasharray="1000"
            strokeDashoffset="0"
          />
          
          {/* SVG icon */}
          <path 
            d="M35 35L65 35M35 50L55 50" 
            stroke="url(#paint1_linear)" 
            strokeWidth="5" 
            strokeLinecap="round"
            strokeDasharray="100"
            strokeDashoffset="0"
          />
          
          <defs>
            <linearGradient 
              id="paint0_linear" 
              x1="15" 
              y1="11" 
              x2="85" 
              y2="70" 
              gradientUnits="userSpaceOnUse"
              gradientTransform="rotate(5)"
            >
              <stop stopColor="#7c3aed" />
              <stop offset="1" stopColor="#0ea5e9" />
            </linearGradient>
            <linearGradient 
              id="paint1_linear" 
              x1="35" 
              y1="35" 
              x2="65" 
              y2="50" 
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="#7c3aed" />
              <stop offset="1" stopColor="#0ea5e9" />
            </linearGradient>
          </defs>
        </svg>
      </Box>
      
      <Box
        component="span"
        sx={{
          fontWeight: 700,
          fontSize: size === 'small' ? '1rem' : size === 'large' ? '1.75rem' : '1.25rem',
          background: 'linear-gradient(90deg, #7c3aed, #0ea5e9)',
          backgroundSize: '200% 200%',
          animation: `${gradientShift} 4s ease infinite`,
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          display: 'inline-block',
        }}
      >
        chat2svg
      </Box>
    </Box>
  );
} 