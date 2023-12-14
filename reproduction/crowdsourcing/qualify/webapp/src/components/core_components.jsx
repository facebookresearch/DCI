/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";

function LoadingScreen() {
  return <Directions>Loading...</Directions>;
}

function Directions({ children }) {
  return (
    <section className="hero is-light" data-cy="directions-container">
      <div className="hero-body">
        <div className="container">
          <p className="subtitle is-5">{children}</p>
        </div>
      </div>
    </section>
  );
}

function SimpleFrontend({ taskData, onSubmit, onError }) {
  const [viewPerspective, setViewPerspective] = React.useState("");
  const [imageCaption, setImageCaption] = React.useState("");
  const [imageExtension, setImageExtension] = React.useState("");
  const wordCount = imageCaption.split(' ').length;
  const disabled = (wordCount < 30) || (viewPerspective.length < 20) || (imageExtension.length < 20)
  const res = {
    viewPerspective, imageCaption, imageExtension
  }
  return (
    <div>
      <Directions>
        Directions: Please fill out the following form!
      </Directions>
      <section className="section">
        <div className="container">
          <p className="subtitle is-5"></p>
          <p className="title is-3 is-spaced" data-cy="task-data-text">
            Imagine you were trying to describe an image to someone not present, such that they would
            be able to accurately sketch the image you're looking at. How would you go about 
            doing this? (~2 sentences)
          </p>
          <div className="field is-grouped">
            <div className="control">
              <textarea
                placeholder='What is important to describe an image?' 
                onChange={(e) => (setViewPerspective(e.target.value))} 
                value={viewPerspective} 
                rows={3}
                cols={80}
              />
            </div>
          </div>
          <p className="title is-3 is-spaced" data-cy="task-data-text">
            Provide a 30-50 word descriptive caption about this image, capturing as much detail as you can in that space.
          </p>
          <img src='truck.jpg' />
          <br />
          Word Count: {wordCount}
          <br />
          <div className="field is-grouped">
            <div className="control">
              <textarea
                placeholder='Describe this image in detail.' 
                onChange={(e) => (setImageCaption(e.target.value))} 
                value={imageCaption} 
                rows={3}
                cols={80}
              />
            </div>
          </div>
          <p className="title is-3 is-spaced" data-cy="task-data-text">
            Imagine you instead had to provide 500-1000 meaningful words about this image as well. What might you consider doing to ensure the
            descriptions are useful, and capture relevant information about what is visually present? (~2 sentences, don't actually provide such a description!)
          </p>
          <div className="field is-grouped">
            <div className="control">
              <textarea
                placeholder='How would you describe the image in even more detail?' 
                onChange={(e) => (setImageExtension(e.target.value))} 
                value={imageExtension} 
                rows={3}
                cols={80}
              />
            </div>
          </div>
          <div className="field is-grouped">
            <div className="control">
              <button
                className="button is-success"
                onClick={() => onSubmit(res)}
                disabled={disabled}
              >
                Submit
              </button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export { LoadingScreen, SimpleFrontend as BaseFrontend };
