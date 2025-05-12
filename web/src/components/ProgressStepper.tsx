import React from 'react';
import {
    Stepper,
    Step,
    StepLabel,
    StepContent,
    Button,
    Typography,
    Box,
    useTheme,
    alpha,
    Paper,
    StepConnector,
    stepConnectorClasses,
    styled
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import PendingIcon from '@mui/icons-material/Pending';

interface ProgressStepperProps {
    currentStage: number;
    failedStage: number | null;
    onRetry: (stage: number) => void;
}

// Custom connector for stepper
const StyledConnector = styled(StepConnector)(({ theme }) => ({
    [`&.${stepConnectorClasses.alternativeLabel}`]: {
        top: 10,
    },
    [`&.${stepConnectorClasses.active}`]: {
        [`& .${stepConnectorClasses.line}`]: {
            background: 'linear-gradient(90deg, #7c3aed, #0ea5e9)',
        },
    },
    [`&.${stepConnectorClasses.completed}`]: {
        [`& .${stepConnectorClasses.line}`]: {
            background: 'linear-gradient(90deg, #7c3aed, #0ea5e9)',
        },
    },
    [`& .${stepConnectorClasses.line}`]: {
        height: 3,
        border: 0,
        backgroundColor: theme.palette.mode === 'dark' ? alpha(theme.palette.grey[800], 0.5) : alpha(theme.palette.grey[300], 0.8),
        borderRadius: 1,
    },
}));

export default function ProgressStepper({currentStage, failedStage, onRetry}: ProgressStepperProps) {
    const theme = useTheme();
    const isDarkMode = theme.palette.mode === 'dark';
    
    const steps = [
        {
            label: 'SVG Template Generation',
            description: 'Generating SVG template from text prompt using LLM',
        },
        {
            label: 'Detail Enhancement',
            description: 'Enhancing SVG details using diffusion models',
        },
        {
            label: 'SVG Optimization',
            description: 'Optimizing SVG for best visual quality',
        },
    ];

    const getStepIcon = (stepIndex: number) => {
        const stepNumber = stepIndex + 1;
        
        if (stepNumber < currentStage) {
            return <CheckCircleIcon sx={{ 
                color: theme.palette.success.main,
                filter: 'drop-shadow(0px 0px 3px rgba(34, 197, 94, 0.4))'
            }} />;
        }
        
        if (failedStage === stepNumber) {
            return <ErrorIcon sx={{ 
                color: theme.palette.error.main,
                filter: 'drop-shadow(0px 0px 3px rgba(239, 68, 68, 0.4))'
            }} />;
        }
        
        if (stepNumber === currentStage) {
            return <PendingIcon sx={{ 
                color: theme.palette.primary.main,
                animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                '@keyframes pulse': {
                    '0%, 100%': {
                        opacity: 1,
                    },
                    '50%': {
                        opacity: 0.5,
                    },
                },
            }} />;
        }
        
        return <div style={{ 
            width: 24, 
            height: 24, 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            borderRadius: '50%', 
            color: isDarkMode ? theme.palette.grey[400] : theme.palette.grey[700],
            border: `2px solid ${isDarkMode ? theme.palette.grey[700] : theme.palette.grey[300]}`,
            backgroundColor: 'transparent',
            fontWeight: 600,
            fontSize: 14
        }}>
            {stepNumber}
        </div>;
    };

    return (
        <Paper 
            elevation={0} 
            sx={{ 
                borderRadius: 2,
                backgroundColor: 'transparent',
                py: 1
            }}
        >
            <Stepper 
                activeStep={Math.max(0, currentStage - 1)} 
                orientation="horizontal" 
                connector={<StyledConnector />}
                sx={{ px: 2 }}
            >
                {steps.map((step, index) => {
                    const stepProps: { completed?: boolean; error?: boolean } = {};
                    const labelProps: { optional?: React.ReactNode } = {};
                    
                    if (index + 1 < currentStage) {
                        stepProps.completed = true;
                    }
                    
                    if (failedStage === index + 1) {
                        stepProps.error = true;
                    }
                    
                    return (
                        <Step key={step.label} {...stepProps}>
                            <StepLabel 
                                icon={getStepIcon(index)}
                                sx={{
                                    '.MuiStepLabel-labelContainer': {
                                        marginTop: 0.5
                                    },
                                    '.MuiStepLabel-label': {
                                        fontWeight: 600,
                                        color: () => {
                                            if (failedStage === index + 1) return theme.palette.error.main;
                                            if (index + 1 === currentStage) return theme.palette.primary.main;
                                            if (index + 1 < currentStage) return theme.palette.success.main;
                                            return isDarkMode ? alpha(theme.palette.text.primary, 0.6) : theme.palette.text.secondary;
                                        }
                                    }
                                }}
                            >
                                {step.label}
                                <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 0.5 }}>
                                    {step.description}
                                </Typography>
                                
                                {failedStage === index + 1 && (
                                    <Box sx={{ mt: 1.5, mb: 0.5 }}>
                                        <Button 
                                            variant="contained" 
                                            color="error" 
                                            onClick={() => onRetry(index + 1)}
                                            sx={{ 
                                                textTransform: 'none',
                                                borderRadius: 1.5,
                                                fontWeight: 600,
                                                boxShadow: 'none',
                                                '&:hover': {
                                                    boxShadow: 'none'
                                                }
                                            }}
                                        >
                                            Retry
                                        </Button>
                                    </Box>
                                )}
                            </StepLabel>
                        </Step>
                    );
                })}
            </Stepper>
        </Paper>
    );
}