/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React, {useRef, useEffect, useState} from "react";


declare type Point = {
  x: number,
  y: number,
}

function MaskOverlay({maskImage, topLeft, bottomRight, imgWidth, imgHeight, height, width }: {
  maskImage: string,
  topLeft: Point,
  bottomRight: Point, 
  imgWidth: number,
  imgHeight: number,
  height: number, // render height
  width: number, // render width
}) {
  const imgTop = topLeft['y'];
  const imgLeft = topLeft['x'];
  const imgBottom = bottomRight['y'];
  const imgRight = bottomRight['x'];
  const cropHeight = imgBottom - imgTop;
  const cropWidth = imgRight - imgLeft;
  const scaleFactorH = height / cropHeight;
  const scaleFactorW = width / cropWidth;
  const scaleFactor = Math.min(scaleFactorH, scaleFactorW);
  const innerHeight = scaleFactor * imgHeight;
  const innerWidth = scaleFactor * imgWidth;

  const left = imgLeft*scaleFactor;
  const top = imgTop*scaleFactor;
  const posString =`${-left}px ${-top}px`;

  return (
    <div>
      <div style={{
        height: height,
        width: width,
        overflow: "hidden",
        opacity: 0.38,
        filter: "drop-shadow(0px 0px 4px rgba(255,255,255,0.75))",
        position: "absolute",
        top: 0,
      }}>
        <div 
          style={{
            backgroundColor: 'cyan', 
            position: 'absolute',
            objectFit: 'cover', 
            objectPosition: posString,
            maskImage: `url(${maskImage})`,
            WebkitMaskImage: `url(${maskImage})`,
            maskSize: `${innerWidth}px ${innerHeight}px`,
            WebkitMaskSize: `${innerWidth}px ${innerHeight}px`,
            maskPosition: posString,
            WebkitMaskPosition: posString,
            height: `${height}px`, 
            width: `${width}px`,
          }}
        />
      </div>
    </div>
  )
}

function WrappedImage({ baseImage, topLeft, bottomRight, height, width}: {
  baseImage: string,
  topLeft: Point,
  bottomRight: Point,
  height: number,
  width: number,
}) {
  const imgTop = topLeft['y'];
  const imgLeft = topLeft['x'];
  const imgBottom = bottomRight['y'];
  const imgRight = bottomRight['x'];
  const cropHeight = imgBottom - imgTop;
  const cropWidth = imgRight - imgLeft;
  const scaleFactorH = height / cropHeight;
  const scaleFactorW = width / cropWidth;
  const scaleFactor = Math.min(scaleFactorH, scaleFactorW);

  return <RawImage
    baseImage={baseImage}
    top={imgTop*scaleFactor}
    left={imgTop*scaleFactor}
    imgHeight={height}
    imgWidth={width}
    height={height}
    width={width}
  />;
}

function RawImage({ baseImage, top, left, imgWidth, imgHeight, height, width }:
  { 
    baseImage: string,
    top: number,
    left: number, 
    imgWidth: number, // required image size to get crop the correct size
    imgHeight: number, // required image size to get crop the correct size
    height: number, // crop render height
    width: number, // crop render width
  }
) {
  const posString =`${-left}px ${-top}px`;

  return (
    <div style={{
      height: height,
      width: width,
      overflow: "hidden",
    }}>
      <img
        src={baseImage}
        height={imgHeight}
        width={imgWidth}
        style={{height: `${imgHeight}px`, width: `${imgWidth}px`, objectFit: 'cover', objectPosition: posString}}
      />
    </div>
  );
}

function DisplayLayer({ baseImage, allMaskData, imgHeight, imgWidth, processedMasks, result }: 
  {
    baseImage: string, 
    allMaskData: any, 
    imgHeight: number, 
    imgWidth: number, 
    processedMasks: any, 
    result: any,
  }
) {
  const [overlayMask, setOverlayMask] = React.useState<null | string>(null);

  const baseCaption = result.baseCaption;
  const finalCaption = result.finalCaption;
  const maskResults = result.data;

  const topLeft = {x: 0, y: 0};
  const bottomRight = {x: imgWidth, y: imgHeight};

  // Dimensions for the image
  const maxWidth = window.innerWidth * 0.65;
  const maxHeight = window.innerHeight * 0.75;
  const width = (bottomRight['x'] - topLeft['x'])
  const height = (bottomRight['y'] - topLeft['y'])
  const widthRatio = width / maxWidth;
  const heightRatio = height / maxHeight;
  let useWidth = width / widthRatio;
  let useHeight = height / heightRatio;
  if (widthRatio > heightRatio) {
    useHeight = height / widthRatio;
  } else if (heightRatio >= widthRatio) {
    useWidth = width / heightRatio;
  }

  const textUseWidth = window.innerWidth - useWidth - 75;
  const textUseHeight = window.innerHeight - 300;

  const previousLabels = Object.keys(allMaskData).filter(
    (reqIdx: string) => allMaskData[reqIdx] !== undefined
  ).map((reqIdx: string) => (
    <span 
      key={`req-${reqIdx}`} 
      onMouseOver={() => setOverlayMask(reqIdx)} 
      onMouseLeave={() => setOverlayMask((oldMask: string | null) => oldMask != reqIdx ? oldMask : null)}
    >
      {
        (maskResults[reqIdx].mask_quality == 0) ? " *" + maskResults[reqIdx].label + "*: " + maskResults[reqIdx].caption
        : (maskResults[reqIdx].mask_quality == 1) ? " *" + maskResults[reqIdx].label + "*: Marked Low qual. "
        : (<i>Bad Mask Skipped</i>)
      }
    </span>
  ))

  return (
    <div>
      <div style={{float: 'right'}} >
        {
          (previousLabels.length == 0) ? null : 
          <span style={{width: textUseWidth, height: textUseHeight, overflowY: 'scroll', display: "block"}}>
            <b>Description:</b> <i>{baseCaption}</i> {finalCaption} <br/> {previousLabels}
          </span>
        }
      </div>
      <div style={{height: useHeight, width: useWidth, position: "relative"}}>
        <WrappedImage 
          baseImage={baseImage}
          topLeft={topLeft}
          bottomRight={bottomRight}
          height={useHeight}
          width={useWidth}
        />
        {(overlayMask == null) ? null : <MaskOverlay
          maskImage={processedMasks[overlayMask]}
          topLeft={topLeft}
          bottomRight={bottomRight}
          imgHeight={imgHeight}
          imgWidth={imgWidth}
          height={useHeight}
          width={useWidth}
        />}
      </div>
      <div style={{clear: 'both'}} />
    </div>
  )
}

function convertMask(imgData: string, width: number, height: number, callback: (processed: string) => void) {
  const canvas = document.createElement('canvas');
  const image = new Image();

  image.onload = function () {
    canvas.width = width;
    canvas.height = height;
    console.log(width, height);
    const ctx = canvas.getContext('2d');
    if (ctx === null) {
      return;
    }
    ctx.drawImage(
      image,
      0,
      0,
      width,
      height,
      0,
      0,
      width,
      height,
    );

    const idata = ctx.getImageData(0, 0, width, height);            // get imagedata
    const data32 = new Uint32Array(idata.data.buffer);     // use uint32 view for speed
    const len = data32.length;
    
    for(let i = 0; i < len; i++) {
      // black pixels become transparent
      const isWhite = data32[i] & 0x00ffffff;
      if (isWhite) {
        data32[i] = 0xFF000000;
      } else {
        data32[i] = 0x00000000;
      }
    }

    // done
    ctx.putImageData(idata, 0, 0);

    // Converting to base64
    const base64Image = canvas.toDataURL('image/png');
    callback(base64Image);
  }
  image.src = "data:image/png;base64," + imgData;
}

function getFullCaption(submitData: any): string {
  let caption = submitData.baseCaption.trim();
  caption += " " + submitData.finalCaption.trim();
  if (!(caption.endsWith('.') || caption.endsWith('!') || caption.endsWith('?'))) {
    caption = caption + '.';
  }
  for (let entryKey in submitData.data) {
    let mask = submitData.data[entryKey]
    if (mask.mask_quality == 0 && mask.label.trim().length > 0) {
      caption += " " + mask.label.trim() + ": " + mask.caption.trim();
    }
    if (!(caption.endsWith('.') || caption.endsWith('!') || caption.endsWith('?'))) {
      caption = caption + '.';
    }
  }
  return caption
}

function currentWordCount(submitData: any): number {
  const currCaption = getFullCaption(submitData);
  return currCaption.split(' ').length;
}

function TaskFrontend({ taskData, result }
  :{
    taskData: any, result: any
  }
) {
  const maskData = taskData.mask_data;
  const imgData = taskData.image_data.image;
  const imgHeight = taskData.image_data.height;
  const imgWidth = taskData.image_data.width;

  const [processedMasks, setProcessedMasks] = React.useState({});
  const [masksLoaded, setMasksLoaded] = React.useState(false);

  React.useEffect(() => {
    for (let maskKey in maskData) {
      const dat = maskData[maskKey]
      convertMask(dat.outer_mask, imgWidth, imgHeight, (newMask) => {
        setProcessedMasks(prevMasks => {return {...prevMasks, [maskKey]: newMask}})})
    }
    setMasksLoaded(true);
  }, [maskData, setProcessedMasks])

  if (!masksLoaded) {
    return <p>... loading masks ...</p>
  }

  const totalMasks = Object.keys(maskData).length

  const wordCount = currentWordCount(result);

  const body = (
    <div style={{padding: '2px'}} >
      <DisplayLayer 
        baseImage={imgData}
        allMaskData={maskData}
        imgHeight={imgHeight} 
        imgWidth={imgWidth} 
        processedMasks={processedMasks} 
        result={result}
      />
    </div>
  );

  return (
    <div tabIndex={0}>
      {body}
      <span>Total Words: {wordCount} Total Masks: {totalMasks}</span>
    </div>
  );
}

export default function ReviewView({ init_data, res }: {init_data: any, res: any}) {
  console.log(init_data, res)
  if (!init_data) {
    return <p>... loading init data ...</p>
  }
  if (!init_data.image_data) {
    return <p>... loading image data ...</p>
  }

  return (
    <TaskFrontend taskData={init_data} result={res} />
  );
}
