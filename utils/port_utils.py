#!/usr/bin/env python3
"""
Port utilities for automatic COM port detection on Windows and Linux/macOS.
"""

import logging
import platform
import os

logger = logging.getLogger(__name__)

def list_available_serial_ports():
    """List all available serial ports"""
    available_ports = []
    
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            available_ports.append({
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid
            })
            
        logger.info(f"Found {len(available_ports)} available serial ports")
        for port in available_ports:
            logger.info(f"  - {port['device']}: {port['description']}")
            
    except ImportError:
        logger.warning("pyserial not available for port detection")
    except Exception as e:
        logger.error(f"Error listing serial ports: {e}")
    
    return available_ports

def find_arduino_ports():
    """Find potential Arduino ports based on description"""
    arduino_ports = []
    
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        
        # Common Arduino identifiers
        arduino_keywords = [
            'arduino', 'ch340', 'ch341', 'cp210x', 'ftdi', 
            'usb serial', 'usb-serial', 'mega', 'uno'
        ]
        
        for port in ports:
            description_lower = port.description.lower()
            hwid_lower = port.hwid.lower() if port.hwid else ""
            
            for keyword in arduino_keywords:
                if keyword in description_lower or keyword in hwid_lower:
                    arduino_ports.append(port.device)
                    logger.info(f"Potential Arduino found: {port.device} ({port.description})")
                    break
                    
    except ImportError:
        logger.warning("pyserial not available for Arduino detection")
    except Exception as e:
        logger.error(f"Error finding Arduino ports: {e}")
    
    return arduino_ports

def find_dynamixel_ports():
    """Find potential Dynamixel controller ports"""
    dynamixel_ports = []
    
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        
        # Common Dynamixel controller identifiers
        dynamixel_keywords = [
            'robotis', 'u2d2', 'usb2dynamixel', 'opencm'
        ]
        
        for port in ports:
            description_lower = port.description.lower()
            hwid_lower = port.hwid.lower() if port.hwid else ""
            
            for keyword in dynamixel_keywords:
                if keyword in description_lower or keyword in hwid_lower:
                    dynamixel_ports.append(port.device)
                    logger.info(f"Potential Dynamixel controller found: {port.device} ({port.description})")
                    break
                    
    except ImportError:
        logger.warning("pyserial not available for Dynamixel detection")
    except Exception as e:
        logger.error(f"Error finding Dynamixel ports: {e}")
    
    return dynamixel_ports

def test_port_connection(port, baudrate=9600, timeout=1.0):
    """Test if a port can be opened"""
    try:
        import serial
        test_port = serial.Serial(port, baudrate, timeout=timeout)
        test_port.close()
        return True
    except Exception:
        return False

def get_recommended_ports():
    """Get recommended port configuration based on available hardware"""
    recommendations = {
        'arduino_ports': [],
        'dynamixel_ports': [],
        'all_available': []
    }
    
    # Get all available ports
    available = list_available_serial_ports()
    recommendations['all_available'] = [port['device'] for port in available]
    
    # Find Arduino ports
    arduino_ports = find_arduino_ports()
    recommendations['arduino_ports'] = arduino_ports
    
    # Find Dynamixel ports
    dynamixel_ports = find_dynamixel_ports()
    recommendations['dynamixel_ports'] = dynamixel_ports
    
    # Generate configuration suggestions
    suggestions = []
    
    if arduino_ports:
        suggestions.append(f"Arduino ports found: {arduino_ports}")
        suggestions.append(f"Suggest setting ARDUINO_PORT = '{arduino_ports[0]}'")
    
    if dynamixel_ports:
        suggestions.append(f"Dynamixel ports found: {dynamixel_ports}")
        if len(dynamixel_ports) >= 2:
            suggestions.append(f"Suggest setting PORT_ACM0 = '{dynamixel_ports[0]}', PORT_ACM1 = '{dynamixel_ports[1]}'")
        elif len(dynamixel_ports) == 1:
            suggestions.append(f"Suggest setting PORT_ACM0 = '{dynamixel_ports[0]}'")
    
    if not arduino_ports and not dynamixel_ports and available:
        suggestions.append("No specific hardware detected, but found generic serial ports:")
        for port in available[:3]:  # Show first 3
            suggestions.append(f"  - {port['device']}: {port['description']}")
    
    recommendations['suggestions'] = suggestions
    
    return recommendations

if __name__ == "__main__":
    """Test port detection when run directly"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    print("Serial Port Detection Tool")
    print("=" * 40)
    print(f"Platform: {platform.system()}")
    
    recommendations = get_recommended_ports()
    
    print(f"\nAvailable Ports ({len(recommendations['all_available'])}):")
    for port in recommendations['all_available']:
        status = "OK" if test_port_connection(port) else "BUSY/ERROR"
        print(f"  {port} - {status}")
    
    print(f"\nArduino Ports ({len(recommendations['arduino_ports'])}):")
    for port in recommendations['arduino_ports']:
        print(f"  {port}")
    
    print(f"\nDynamixel Ports ({len(recommendations['dynamixel_ports'])}):")
    for port in recommendations['dynamixel_ports']:
        print(f"  {port}")
    
    if recommendations['suggestions']:
        print("\nRecommendations:")
        for suggestion in recommendations['suggestions']:
            print(f"  {suggestion}")
    else:
        print("\nNo hardware-specific ports detected.")
    
    print("\nTo use these ports, update your config.py file accordingly.")