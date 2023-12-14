/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React, {useRef, useEffect, useState} from "react";
import {Button, ButtonToolbar, Tooltip, OverlayTrigger, InputGroup, FormControl} from "react-bootstrap";
import ReviewComponent from "./review_component"

function LoadingScreen() {
  return <Directions>Loading...</Directions>;
}

function Directions({ children }: {children: any}) {
  return (
    <section className="hero is-light">
      <div className="hero-body">
        <div className="container">
          <p className="subtitle is-5"><b>Directions: </b>{children}</p>
        </div>
      </div>
    </section>
  );
}

function Instructions() {
  return (
    <div style={{backgroundColor: 'lightgray', marginLeft: "30px", maxWidth: '1200px', display: "block"}}>
      <h3>Summary</h3>
      <p>
        The goal of this task is to annotate as much visual information as is present in the given 
        image into text, <b>such that it would be possible to create a (nearly) identical image just 
        from your text.</b> To accomplish this goal, you'll walk through a single image in stages, building
        up from small details in the image one may normally overlook into description larger components.
        We expect the task to take around 30-40 minutes per image.
      </p>
      <p>
        We'll combine all of the text into a rough description of the image, and be storing all of the
        annotations provided. Let's walk through an example.
      </p>
      <br/>
      <h3>Step 1: Base Caption</h3>
      <p>
        Initially you'll be given a complete image and asked to provide a simple two sentence caption
        for what you see. This doesn't need to be highly detailed, 20-30 words is plenty.
      </p>
      <img src="/01-full-image.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <br/>
      <h3>Step 2: Building up</h3>
      <p>
        Most of the task will be in this stage. We use a model to segment the image into components 
        for you to annotate. You will be shown a segment of the image on the left <b>with a specific
        highlighted region</b> to describe. The goal is not to annotate the entire rectangular crop shown,
        but just the highlighted region inside. On the right will display where in the full image the crop
        is located. In providing annotations here, we're looking for short labels and then
        longer descriptions capturing as much visual detail as possible. 
      </p>
      <p>
        As these segments are automatically generated, they're not always useful to describe in detail,
        or at times even accurate. For instance, if you're posed with something low detail that would
        not be useful to describe, you can press "low quality", provide a label, and move on to the 
        next mask.
      </p>
      <img src="/02-low-qual.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <p>
        Sometimes it can be hard to come up with what to provide for a description, remember that details
        about where something is located or how it is laid out can be important.
      </p>
      <img src="/03-describe-location.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <p>
        Sometimes a section of image is rather large, but still only has a sentence or so to share
        for the full description. This is ok, so long as it captures as much descriptive detail as
        you can provide.
      </p>
      <img src="/04-large-bg.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <p>
        Do your best to capture details whenever the model provides you with a new region, focusing only on
        describing the highlighted area.
      </p>
      <img src="/05-best-effort.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <p>
        Some regions are relatively small for the image, but still can provide large amounts of detailed 
        description. Do your best to capture the most important details you can see in each of these regions.
      </p>
      <img src="/06-more-detail.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <p>
        Some regions may have a lot for you to describe. These can include elements of color, layout, shape, etc.
        Do try to pace yourself, as ultimately we're aiming for around 750 words.
      </p>
      <img src="/07-more-descript.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <p>
        Eventually, regions will overap with smaller regions you've already annotated. There's no need to
        reiterate details you've already described, so focus on new details or how the sub-regions are laid out.
        You can see what sub regions were included by hovering over the text to the right of the image.
      </p>
      <img src="/08-no-reiterate.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <p>
        Some provided regions may be simply wrong, splitting something in the image in a way that doesn't 
        make sense. Use "Mark Bad" to move on past these masks.
      </p>
      <img src="/09-bad-mask.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <p>
        In some cases, it may be useful to reference terms for things you may know are named, but don't know
        the name of (for instance the 'hull' of a boat). This isn't necessary, but may make it easier for
        you to describe things.
      </p>
      <img src="/10-terms-text.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <br/>
      <h3>Step 3: Final Description</h3>
      <p>
        Once you have annotated all the masks, you will be asked once again to provide a description of the
        full image. Here you should try to describe the image in terms of the regions you have annotated (still 
        no need to re-iterate details you've covered), and include details that may not have come up in the
        first pass. This is also an opportunity to be sure to reach the target description length.
      </p>
      <img src="/11-final-description.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <p>
        The final description for the above image could be something like: <i>"This is a photo of a tugboat 
        pushing a barge upstream, taken midday. Behind the tugboat you can see a small passenger boat making 
        a long wake parallel to the shoreline. In that wake is a small jetski. On the shore you can see a 
        large paved area, possibly a parking lot, nearly directly behind where the barge is. On it are scattered 
        orange cones and some yellow barriers. Level with the skyline is a large apartment building to the left 
        of the image, then a series of smaller buildings mostly obscured by trees, and then a large cluster of 
        buildings on the very right side. A warehouse is visible on the right edge of the shoreline with a tugboat 
        parked up nearby. In the background the sky is a light grayish-blue, with some light cumulus clouds.</i>
      </p>
      <p>
        After completing this, you can submit the task, or use the previous/next buttons to check your work.
      </p>
      <br/>
      <h3>Extended Notes & Examples</h3>
      <h4>Specificity</h4>
      <p>
        Different images and regions may require different descriptions. When annotating a region however,
        your focus should be entirely on the content of that region, and perhaps how it directly relates
        to its surroundings.
      </p>
      <img src="/12-roof-details.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <h4>Contained Elements</h4>
      <p>
        When one element is contained within another element, it's important to try to focus on describing
        new information, rather than reiterating. If the only thing possible is the same description, it
        may be reasonable to go back to the prior element, mark it as bad, and copy your description to the
        next element.
      </p>
      <p>
        Take the following complex example, and note how subsequent descriptions usually only mention the 
        label of one of the contained areas, but don't reiterate the same details. Instead, they focus on
        new information
      </p>
      <img src="/13-subImage-umbrellas.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <img src="/14-detail-window-1.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <img src="/15-detail-window-2.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <h4>Unidentifiable Content</h4>
      <p>
        Not every mask is going to be identifyable or high-quality. If there's no way at all to tell
        what something is, use the "Mark Bad" button. If a label is still possible, but there's very
        little else that is useful to describe, use "Low Quality". 
      </p>
      <img src="/16-low-qual-simple.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <p>
        In cases that are less clear, you can use your judgement on which label to be providing.
      </p>
      <img src="/17-unclear.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <h4>Transcription</h4>
      <p>
        If an image only has a small quantity of visible words, it is useful to include anything
        you can make out. If an image is almost entirely text, but there are more interesting things
        to describe, don't feel the need to get every single word noted.
      </p>
      <img src="/18-text-transcription.png" style = {{maxWidth: '1100px', paddingBottom: "12px", marginLeft: "20px"}}/>
      <h4>Editing</h4>
      <p>
        If at any point you feel you want to make changes to previous descriptions you can go
        back using the "Previous Step" button. This will allow you to make changes, which can be useful
        if you're wanting to extend previous descriptions, change labels based on future masks, or
        refine your work.
      </p>
      <h4>Word Count</h4>
      <p>
        Our target words per image are around 750 words. Use your discretion for how to distribute
        those words throughout the image. You can go over, and at times you can go under, but note
        that doing so when there was certainly more to describe can lead to getting soft blocked
      </p>
      <h4>Efficiency + Keyboard Shortcuts</h4>
      <p>
        In order to make the task efficient, we've made it possible to execute entirely from the 
        keyboard. You can use TAB or SHIFT+TAB to switch between inputs, ENTER to advance to the 
        next step, SHIFT+ENTER for the previous step, ESC to mark as low quality, and SHIFT+ESC to 
        mark a mask as bad.
      </p>
    </div>
  )
}

function InstructionContainer() {
  let [isHidden, setHidden] = React.useState(true);
  return (
    <>
      <div>
        <button onClick={() => {setHidden(!isHidden)}}>{(isHidden) ? "Show Instructions" : "Hide Instructions"}</button>
      </div>
      {(isHidden) ? null : <Instructions/>}
    </>
  )
}

declare type Point = {
  x: number,
  y: number,
}

function MaskRenderer({baseImage, maskImage, topLeft, bottomRight, imgWidth, imgHeight, height, width }: {
  baseImage: string,
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

  return (
    <div>
      <div style={{height, width, backgroundColor: 'black', position: 'absolute'}} ></div>
      <div style={{opacity: 0.5, position: 'absolute'}}>
        <RawImage
          baseImage={baseImage}
          top={imgTop*scaleFactor}
          left={imgLeft*scaleFactor}
          imgWidth={innerWidth}
          imgHeight={innerHeight}
          height={height}
          width={width}
        />
      </div>
      <div style={{position:'absolute'}}>
        <MaskedImage
          baseImage={baseImage}
          maskImage={maskImage}
          top={imgTop*scaleFactor}
          left={imgLeft*scaleFactor}
          imgWidth={innerWidth}
          imgHeight={innerHeight}
          height={height}
          width={width}
        />
      </div>
    </div>
  )
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

function MaskBBoxView({baseImage, topLeft, bottomRight, imgWidth, imgHeight, height, width }:
  { 
    baseImage: string,
    topLeft: Point,
    bottomRight: Point, 
    imgWidth: number,
    imgHeight: number,
    height: number, // render height
    width: number, // render width
  }
) {

  const imgTop = topLeft['y'];
  const imgLeft = topLeft['x'];
  const imgBottom = bottomRight['y'];
  const imgRight = bottomRight['x'];
  const scaleFactorH = height / imgHeight;
  const scaleFactorW = width / imgWidth;
  const scaleFactor = Math.min(scaleFactorH, scaleFactorW);
  const rectHeight = (imgBottom - imgTop) * scaleFactor;
  const rectWidth = (imgRight - imgLeft) * scaleFactor;
  const rectTop = imgTop * scaleFactor;
  const rectLeft = imgLeft * scaleFactor;

  return (
    <div>
      <img
        src={baseImage}
        height={height}
        width={width}
        style={{
          objectFit: 'cover',
          height: height,
          width: width,
        }}
      />
      <div
          style={{
              border: `3px solid rgb(255,0,0)`,
              position: 'absolute',
              top: `${rectTop}px`,
              left: `${rectLeft}px`,
              width: `${rectWidth}px`,
              height: `${rectHeight}px`,
              color: 'rgb(255,0,0)',
              fontFamily: 'monospace',
              fontSize: 'small',
          }}
      />
    </div>
  );
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

function MaskedImage({ baseImage, maskImage, top, left, imgWidth, imgHeight, height, width }:
  { 
    baseImage: string,
    maskImage: string,
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
      filter: "drop-shadow(0px 0px 4px rgba(195,255,255,0.95))"
    }}>
      <img
        src={baseImage}
        height={imgHeight}
        width={imgWidth}
        style={{
          objectFit: 'cover', 
          objectPosition: posString,
          maskImage: `url(${maskImage})`,
          WebkitMaskImage: `url(${maskImage})`,
          maskSize: `${imgWidth}px ${imgHeight}px`,
          WebkitMaskSize: `${imgWidth}px ${imgHeight}px`,
          maskPosition: posString,
          WebkitMaskPosition: posString,
          height: `${imgHeight}px`, 
          width: `${imgWidth}px`,
        }}
      />
      <div
        style={{
          objectFit: 'cover', 
          objectPosition: posString,
          maskImage: `url(${maskImage})`,
          WebkitMaskImage: `url(${maskImage})`,
          maskSize: `${imgWidth}px ${imgHeight}px`,
          WebkitMaskSize: `${imgWidth}px ${imgHeight}px`,
          maskPosition: posString,
          WebkitMaskPosition: posString,
          height: `${imgHeight}px`, 
          width: `${imgWidth}px`,
          position: "absolute",
          top: "0px",
          left: "0px",
        }}
      >
        <div style={{
          backgroundColor: "white",
          position: "absolute",
          height: `${height}px`, 
          width: `${width}px`,
          top: "0px",
          left: "0px",
        }} className="shimmer" />
      </div>
    </div>
  );
}

function ComponentLayer({ baseImage, allMaskData, target, imgHeight, imgWidth, processedMasks, addProcessedMask, progress }: 
  {
    baseImage: string, 
    allMaskData: any, 
    target: number | string, 
    imgHeight: number, 
    imgWidth: number, 
    processedMasks: any, 
    addProcessedMask: (newMask: string) => void,
    progress: string,
  }
) {
  const [overlayMask, setOverlayMask] = React.useState(null);
  const targetMask = allMaskData[target];

  // Select bounds, either the parent or at most 3x the height/width of the inner mask
  const parentId = targetMask['parent'];
  let topLeft = null;
  let bottomRight = null
  if (parentId != -1) {
    const firstParent = allMaskData[parentId];
    topLeft = {...firstParent.bounds.topLeft};
    bottomRight = {...firstParent.bounds.bottomRight};
  } else {
    topLeft = {x: 0, y: 0};
    bottomRight = {x: imgWidth, y: imgHeight};
  }

  const innerTopLeft = targetMask.bounds.topLeft;
  const innerBottomRight = targetMask.bounds.bottomRight;
  const innerWidth = innerBottomRight.x - innerTopLeft.x;
  const innerHeight = innerBottomRight.y - innerTopLeft.y;
  const minX = innerTopLeft.x - innerWidth * 0.6;
  const maxX = innerBottomRight.x + innerWidth * 0.6;
  const minY = innerTopLeft.y - innerHeight * 0.3;
  const maxY = innerBottomRight.y + innerHeight * 0.3;
  topLeft.x = Math.max(minX, topLeft.x);
  topLeft.y = Math.max(minY, topLeft.y);
  bottomRight.x = Math.min(maxX, bottomRight.x);
  bottomRight.y = Math.min(maxY, bottomRight.y);

  const maskString = "data:image/png;base64," + targetMask.outer_mask;

  React.useEffect(() => {
    convertMask(maskString, imgWidth, imgHeight, addProcessedMask)
  }, [maskString, imgWidth, imgHeight])

  if (processedMasks[target] === undefined) {
    return <p>... loading mask ...</p>
  }

  // Dimensions for the bbox viewer
  const maxWidth = window.innerWidth * 0.29;
  const maxHeight = window.innerHeight * 0.75;
  const widthRatio = imgWidth / maxWidth;
  const heightRatio = imgHeight / maxHeight;
  let useWidth = imgWidth / widthRatio;
  let useHeight = imgHeight / heightRatio;
  if (widthRatio > heightRatio) {
    useHeight = imgHeight / widthRatio;
  } else if (heightRatio >= widthRatio) {
    useWidth = imgWidth / heightRatio;
  }

  // Dimensions for the mask viewer
  const maxMaskWidth = window.innerWidth * 0.69;
  const maxMaskHeight = window.innerHeight * 0.75;
  const maskWidth = (bottomRight['x'] - topLeft['x'])
  const maskHeight = (bottomRight['y'] - topLeft['y'])
  const maskWidthRatio = maskWidth / maxMaskWidth;
  const maskHeightRatio = maskHeight / maxMaskHeight;
  let maskUseWidth = maskWidth / maskWidthRatio;
  let maskUseHeight = maskHeight / maskHeightRatio;
  if (maskWidthRatio > maskHeightRatio) {
    maskUseHeight = maskHeight / maskWidthRatio;
  } else if (maskHeightRatio >= maskWidthRatio) {
    maskUseWidth = maskWidth / maskHeightRatio;
  }

  const previousLabels = targetMask.requirements.filter(
    (reqIdx: string) => allMaskData[reqIdx] !== undefined
  ).map((reqIdx: string) => (
    <span 
      key={`req-${reqIdx}`} 
      onMouseOver={() => setOverlayMask(reqIdx)} 
      onMouseLeave={() => setOverlayMask((oldMask: string) => oldMask != reqIdx ? oldMask : null)}
    >
      {allMaskData[reqIdx].label + ", "}
    </span>
  ))

  return (
    <div>
      <div style={{float: 'right'}} >
        <span>Location in image:</span>
        <div style={{
          height: useHeight, 
          width: useWidth, 
          position: "relative",
        }}>
          <MaskBBoxView
            baseImage={baseImage}
            imgHeight={imgHeight}
            imgWidth={imgWidth}
            height={useHeight}
            width={useWidth}
            topLeft={topLeft}
            bottomRight={bottomRight}
          />
        </div>
        {
          (previousLabels.length == 0) ? null : 
          <span style={{width: useWidth, display: "block"}}>
            This mask is comprised of prior layers: {previousLabels} (hover text to highlight)
          </span>
        }
      </div>
      <span><b>Current Mask Region:</b> (highlighted) {progress}</span>
      <div style={{height: maskUseHeight, width: maskUseWidth, position: "relative"}}>
        <MaskRenderer 
          baseImage={baseImage}
          maskImage={processedMasks[target]}
          topLeft={topLeft}
          bottomRight={bottomRight}
          imgHeight={imgHeight}
          imgWidth={imgWidth}
          height={maskUseHeight}
          width={maskUseWidth}
        />
        {(overlayMask == null) ? null : <MaskOverlay
          maskImage={processedMasks[overlayMask]}
          topLeft={topLeft}
          bottomRight={bottomRight}
          imgHeight={imgHeight}
          imgWidth={imgWidth}
          height={maskUseHeight}
          width={maskUseWidth}
        />}
      </div>
      <div style={{clear: 'both'}} />
    </div>
  )
}

function FinalLayer({ baseImage, allMaskData, imgHeight, imgWidth, processedMasks }: 
  {
    baseImage: string, 
    allMaskData: any, 
    imgHeight: number, 
    imgWidth: number, 
    processedMasks: any, 
  }
) {
  const [overlayMask, setOverlayMask] = React.useState(null);

  const topLeft = {x: 0, y: 0};
  const bottomRight = {x: imgWidth, y: imgHeight};

  // Dimensions for the image
  const maxWidth = window.innerWidth * 0.8;
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

  const textUseWidth = window.innerWidth - useWidth - 25;

  const previousLabels = Object.keys(allMaskData).filter(
    (reqIdx: string) => allMaskData[reqIdx] !== undefined && allMaskData[reqIdx]['parent'] == -1
  ).map((reqIdx: string) => (
    <span 
      key={`req-${reqIdx}`} 
      onMouseOver={() => setOverlayMask(reqIdx)} 
      onMouseLeave={() => setOverlayMask((oldMask: string) => oldMask != reqIdx ? oldMask : null)}
    >
      {allMaskData[reqIdx].label + ", "}
    </span>
  ))

  return (
    <div>
      <div style={{float: 'right'}} >
        {
          (previousLabels.length == 0) ? null : 
          <span style={{width: textUseWidth, display: "block"}}>
            You've noted the previous layers: {previousLabels} (hover text to highlight)
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


function TaskButtons({ maskInfo, onNext, onPrev, onChangeMaskInfo, isSubmit }: {
  maskInfo: any,
  onNext: () => void,
  onPrev: () => void,
  onChangeMaskInfo: (newInfo: any) => void,
  isSubmit: boolean,
}) {
  function makeLabelButton(text: string) {
    return (
      <Tooltip id="tooltip">
        {text}
      </Tooltip>
    );
  }

  const infoIsValid = (
    (maskInfo == null) ||
    (maskInfo.mask_quality == 2) || 
    (maskInfo.mask_quality == 1 && maskInfo.label.length) || 
    (maskInfo.label.length > 0 && maskInfo.caption.length > 0)
  )

  const lastButton = (
    <OverlayTrigger placement="top" overlay={makeLabelButton("Shortcut: Enter")}>
      <Button disabled={!infoIsValid} onClick={onNext} variant="primary" size="sm">Next Step</Button>
    </OverlayTrigger>
  )
  if (isSubmit) {
    <OverlayTrigger placement="top" overlay={makeLabelButton("Shortcut: Enter")}>
      <Button onClick={onNext} variant="primary" size="sm">Submit Task</Button>
    </OverlayTrigger>
  }

  return (
    <ButtonToolbar>
      <OverlayTrigger placement="top" overlay={makeLabelButton(
        "Mark that a mask is of poor quality, not quite capturing a contained part of the image. Shortcut: Esc"
      )}>
        <Button 
          disabled={isSubmit || maskInfo == null}
          onClick={() => onChangeMaskInfo({...maskInfo, mask_quality: ((maskInfo.mask_quality == 1) ? 0 : 1)})}
          variant="warning"
          size="sm" 
        >
          Low Quality
        </Button>
      </OverlayTrigger>
      <OverlayTrigger placement="top" overlay={makeLabelButton(
        "Note that this mask doesn't show a discernable part of the image. Shortcut: Shift+Esc"
      )}>
        <Button 
          disabled={isSubmit || maskInfo == null}
          onClick={() => onChangeMaskInfo({...maskInfo, mask_quality: ((maskInfo.mask_quality == 2) ? 0 : 2)})}
          variant="danger"
          size="sm" 
        >
          Mark Bad
        </Button>
      </OverlayTrigger>
      <OverlayTrigger placement="top" overlay={makeLabelButton("Shortcut: Shift+Enter")}>
        <Button variant="secondary" size="sm" onClick={onPrev}>Previous Step</Button>
      </OverlayTrigger>
      {lastButton}
    </ButtonToolbar>
  );
}

function TaskFields({ maskInfo, onChangeMaskInfo }: {
  maskInfo: any,
  onChangeMaskInfo: (newInfo: any) => void,
}) {
  let [lastMask, setLastMask] = React.useState(maskInfo.idx);
  let inputRef = React.useRef(null);
  let maskQuality = null;
  if (maskInfo.mask_quality == 1) {
    maskQuality = <i>Mask Marked Low-quality</i>;
  } else if (maskInfo.mask_quality == 2) {
    maskQuality = <strong>Mask Marked Unusable</strong>;
  }

  React.useEffect(() => {
    if (maskInfo.idx != lastMask) {
      inputRef.current.focus();
      setLastMask(maskInfo.idx);
    }
  }, [maskInfo])
  return (
    <div>
      {maskQuality}
      <InputGroup>
        <InputGroup.Prepend><InputGroup.Text>Label:</InputGroup.Text></InputGroup.Prepend>
        <FormControl 
          as="input" 
          placeholder="a few words describing this region of the image" 
          value={maskInfo.label}
          onChange={(e) => {onChangeMaskInfo({...maskInfo, label: e.target.value})}}
          disabled={maskInfo.mask_quality == 2}
          autoFocus
          ref={inputRef}
        />
      </InputGroup>
      <InputGroup>
      <InputGroup.Prepend><InputGroup.Text>Description:</InputGroup.Text></InputGroup.Prepend>
        <FormControl 
          as="textarea" 
          placeholder="A long-text description of this portion of the image." 
          value={maskInfo.caption}
          onChange={(e) => {onChangeMaskInfo({...maskInfo, caption: e.target.value})}}
          disabled={maskInfo.mask_quality != 0}
        />
      </InputGroup>
    </div>
  );
}

function convertMask(imgData: string, width: number, height: number, callback: (processed: string) => void) {
  const canvas = document.createElement('canvas');
  const image = new Image();

  image.onload = function () {
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

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
  image.src = imgData;
}

type TaskState = 'first' | 'masks' | 'last' | 'finished';

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
  console.log(currCaption)
  return currCaption.split(' ').length;
}

function TaskFrontend({ initialTaskData, onSubmit, finalResults = null }
  :{
    initialTaskData: any, onSubmit: any, finalResults: any
  }
) {
  const taskData = initialTaskData;
  if (!taskData) {
    return <p>... loading taskData ...</p>
  }
  if (!taskData.image_data) {
    return <p>... loading image data...</p>
  }
  if (finalResults !== null) {
    return <ReviewComponent init_data={taskData} res={finalResults} />;
  }
  const baseMaskData = taskData.mask_data;

  const [processedMasks, setProcessedMasks] = React.useState({});
  const [maskData, setMaskData] = React.useState(baseMaskData);
  const [currIdx, setCurrIdx] = React.useState(-1);
  const [overallCaption, setOverallCaption] = React.useState("");
  const [finalCaption, setFinalCaption] = React.useState("");
  let divRef = React.useRef(null);

  const idxMapping = Object.entries(maskData).map(
    ([k, v], i) => k
  ).reverse();

  const totalMasks = Object.keys(baseMaskData).length
  let taskState: TaskState = 'masks';
  if (currIdx < 0) {
    taskState = 'first';
  } else if (currIdx == totalMasks) {
    taskState = 'last';
  } else if (currIdx > totalMasks) {
    taskState = 'finished';
  }

  const imgData = taskData.image_data.image;
  const imgHeight = taskData.image_data.height;
  const imgWidth = taskData.image_data.width;

  let maxWidth = window.innerWidth * 0.8;
  let maxHeight = window.innerHeight * 0.75;
  let widthRatio = imgWidth / maxWidth;
  let heightRatio = imgHeight / maxHeight;
  let useWidth = imgWidth / widthRatio;
  let useHeight = imgHeight / heightRatio;
  if (widthRatio > heightRatio) {
    useHeight = imgHeight / widthRatio;
  } else if (heightRatio >= widthRatio) {
    useWidth = imgWidth / heightRatio;
  }

  const submitData = {
    'baseCaption': overallCaption,
    'finalCaption': finalCaption,
    'data': Object.fromEntries(Object.keys(maskData).map(x => [x, 
      {
        'mask_quality': maskData[x].mask_quality,
        'label': maskData[x].label,
        'caption': maskData[x].caption,
      }
    ]))
  };
  const wordCount = currentWordCount(submitData);

  let body = null;
  if (taskState == 'first') {
    body = (
      <div style={{padding: '2px'}} >
        <Directions>
          You will be annotating the below image in stages. Start off with a short (1-2 sentence) caption.
        </Directions>
        <div style={{height: useHeight, width: useWidth, position: "relative"}}>
          <WrappedImage 
            baseImage={imgData}
            topLeft={{x: 0, y: 0}}
            bottomRight={{x: imgWidth, y: imgHeight}}
            height={useHeight}
            width={useWidth}
          />
        </div>
        <InputGroup>
          <InputGroup.Prepend><InputGroup.Text>Caption:</InputGroup.Text></InputGroup.Prepend>
          <FormControl 
            autoFocus
            as="textarea" 
            placeholder="A 1-2 sentence caption for this image." 
            value={overallCaption}
            onChange={(e) => {setOverallCaption(e.target.value)}}
          />
        </InputGroup>
        <Button onClick={() => setCurrIdx(0)} variant="primary" size="sm">Next Step</Button>
      </div>
    );
  } else if (taskState == 'masks') {
    body = (
      <div style={{padding: '2px'}} >
        {(currIdx == 0) ? 
          <Directions>
            Now annotate masks for this image. You are shown a region at a time and will build up to the full image. Provide descriptions
            with as much detail as you can about individual masks, in terms of what you can see and the smaller masks you've already done.
          </Directions> : null
        }
        <ComponentLayer
          baseImage={imgData} 
          allMaskData={maskData}
          target={idxMapping[currIdx]}
          imgHeight={imgHeight} 
          imgWidth={imgWidth} 
          processedMasks={processedMasks} 
          addProcessedMask={(newMask) => setProcessedMasks(prevMasks => {return {...prevMasks, [idxMapping[currIdx]]: newMask}})}
          progress={`Current Mask: (${currIdx+1}/${totalMasks})`}
        />
        <TaskFields 
          maskInfo={maskData[idxMapping[currIdx]]}
          onChangeMaskInfo={(newMaskData) => setMaskData((prevMaskData: any) => {return {...prevMaskData, [idxMapping[currIdx]]: newMaskData}})}
        />
        <TaskButtons 
          maskInfo={maskData[idxMapping[currIdx]]}
          onNext={() => setCurrIdx(currIdx + 1)}
          onPrev={() => setCurrIdx(currIdx - 1)}
          onChangeMaskInfo={(newMaskData) => setMaskData((prevMaskData: any) => {return {...prevMaskData, [idxMapping[currIdx]]: newMaskData}})}
          isSubmit={false}
        />
      </div>
    );
  } else if (taskState == 'last') {
    body = (
      <div style={{padding: '2px'}} >
        <Directions>
          Returning to the full image, annotate with as much additional new detail as you can. No need to repeat previous descriptions, but try to 
          incorporate details about layout, overall scene, and anything else that might have been missing.
        </Directions>
        <FinalLayer 
          baseImage={imgData}
          allMaskData={maskData}
          imgHeight={imgHeight} 
          imgWidth={imgWidth} 
          processedMasks={processedMasks} 
        />
        <InputGroup>
          <InputGroup.Prepend><InputGroup.Text>Description:</InputGroup.Text></InputGroup.Prepend>
          <FormControl 
            autoFocus
            as="textarea" 
            placeholder="A full description for this image." 
            value={finalCaption}
            onChange={(e) => {setFinalCaption(e.target.value)}}
          />
        </InputGroup>
        <TaskButtons 
          maskInfo={null}
          onNext={() => setCurrIdx(currIdx + 1)}
          onPrev={() => setCurrIdx(currIdx - 1)}
          onChangeMaskInfo={(newMaskData) => setMaskData((prevMaskData: any) => {return {...prevMaskData, [idxMapping[currIdx]]: newMaskData}})}
          isSubmit={false}
        />
      </div>
    );
  } else {
    body = (
      <div style={{padding: '5px'}} >
        <div style={{height: useHeight, width: useWidth, position: "relative"}}>
          <WrappedImage 
            baseImage={imgData}
            topLeft={{x: 0, y: 0}}
            bottomRight={{x: imgWidth, y: imgHeight}}
            height={useHeight}
            width={useWidth}
          />
        </div>
        <Directions>
          You are done with this image. Please submit when you're ready.
        </Directions>
        <TaskButtons 
          maskInfo={maskData[idxMapping[currIdx]]}
          onNext={() => {
            onSubmit(submitData)
          }}
          onPrev={() => setCurrIdx(currIdx - 1)}
          onChangeMaskInfo={(newMaskData) => setMaskData((prevMaskData: any) => {return {...prevMaskData, [idxMapping[currIdx]]: newMaskData}})}
          isSubmit={true}
        />
      </div>
    );
  }
  
  const maskInfo = maskData[idxMapping[currIdx]];
  const infoIsValid = (
    (maskInfo == null) ||
    (maskInfo.mask_quality == 2) || 
    (maskInfo.mask_quality == 1 && maskInfo.label.length > 0) || 
    (maskInfo.label.length > 0 && maskInfo.caption.length > 0 )
  )

  function handleKeyDown(event: any) {
    if (event.key == 'Enter') {
      if (!event.shiftKey) {
        if (infoIsValid && taskState != 'finished') {
          setCurrIdx(currIdx + 1);
        }
      } else if (currIdx != -1) {
        let newCurrIdx = currIdx - 1;
        setCurrIdx(newCurrIdx);
        if (newCurrIdx != -1) {
          if (maskData[idxMapping[newCurrIdx]].mask_quality == 2) {
            divRef.current.focus();
          }
        }
      }
      event.preventDefault();
      event.stopPropagation();
    } else if (event.key == 'Escape' && taskState == 'masks') {
      if (event.shiftKey && maskInfo.mask_quality != 2) {
        setMaskData((prevMaskData: any) => {return {...prevMaskData, [idxMapping[currIdx]]: {...maskInfo, mask_quality: 2}}});
        divRef.current.focus();
      } else if (!event.shiftKey && maskInfo.mask_quality != 1) {
        setMaskData((prevMaskData: any) => {return {...prevMaskData, [idxMapping[currIdx]]: {...maskInfo, mask_quality: 1}}});
      } else {
        setMaskData((prevMaskData: any) => {return {...prevMaskData, [idxMapping[currIdx]]: {...maskInfo, mask_quality: 0}}});
      }
      event.preventDefault();
      event.stopPropagation();
    }
  }

  return (
    <div tabIndex={0} onKeyDown={(e) => handleKeyDown(e)} style={{position: 'relative'}} ref={divRef}>
      {body}
      <span style={{position: 'absolute', bottom: '0px', right: '0px'}}>Total Words: {wordCount} / 750 (target)</span>
      <InstructionContainer />
    </div>
  );
}

export { LoadingScreen, TaskFrontend as BaseFrontend, Instructions };
