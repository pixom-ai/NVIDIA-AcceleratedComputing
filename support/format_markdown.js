/**
 * Created by abduld on 3/16/16.
 */
"use strict";

const fs = require('fs');
const path = require('path');
const glob = require('glob');
const tidyMarkdown = require('tidy-markdown');


const teachingKitPath = path.join(__dirname, '..');


glob(path.join(teachingKitPath, '**', '*'), function (er, files) {
  const documents = files.filter((elem) => elem.match(/description\.markdown/));
  documents.map(function (document) {
    fs.readFile(document, 'utf8', function(err, md) {
      console.log("processing " + document);
      const tmd = tidyMarkdown(md);
      fs.writeFile(document, tmd, 'utf8', function(err) {
        if (err) {
          console.log("X failed to tidy " + document);
        } else {
          console.log("O tidy " + document);
        }
      })
    })
  });
});