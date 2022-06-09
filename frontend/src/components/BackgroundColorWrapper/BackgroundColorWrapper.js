import React, { useState } from 'react';
import {
  BackgroundColorContext,
  backgroundColors
} from '../../contexts/BackgroundColorContext';

export default function BackgroundColorWrapper(props) {
  const [color, setColor] = useState(backgroundColors.blue);

  function changeColor(_color) {
    setColor(_color);
  }

  return (
    <BackgroundColorContext.Provider value={{ color, changeColor }}>
      {props.children}
    </BackgroundColorContext.Provider>
  );
}
