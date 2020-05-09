// Import React FilePond
import React from "react";
import { FilePond, registerPlugin } from "react-filepond";
import ReactDOM from "react-dom";
// Import FilePond styles
import "filepond/dist/filepond.min.css";

import Result from "../../result/result.json"

// Import the Image EXIF Orientation and Image Preview plugins
// Note: These need to be installed separately
import FilePondPluginImageExifOrientation from "filepond-plugin-image-exif-orientation";
import FilePondPluginImagePreview from "filepond-plugin-image-preview";
import "filepond-plugin-image-preview/dist/filepond-plugin-image-preview.css";

// Register the plugins
registerPlugin(FilePondPluginImageExifOrientation, FilePondPluginImagePreview);

// Our app
class App extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      // Set initial files, type 'local' means this is a file
      // that has already been uploaded to the server (see docs)
      files: [
        {
          source: " ",
          options: {
            type: "local"
          }
        }
      ]
    };
  }

  handleInit() {
    console.log("FilePond instance has initialised", this.pond);
  }

  render() {
         let title =["image_PATH","image_content"];
          let headTDs = title.map(function(cName,i){
        return <th key = {'col'+i}> {cName}</th>
    });
      
       let contextTD = Result.map((elements,i)=>
        
         <tr key ={'col1_'+i} >
            <td  key ={'col2_'+i}>{elements.image_path}</td>
        <td  key ={'col3_'+i}>{elements.image_content}</td>
       
        </tr>
    );
    
      
      
    return (
      <div className="App">
        {/* Pass FilePond properties as attributes */}
        <FilePond
          ref={ref => (this.pond = ref)}
          files={this.state.files}
          allowMultiple={true}
          maxFiles={3}
         itemInsertLocation='before'
         server="http://127.0.0.1:3000/upload"
          oninit={() => this.handleInit()}
          onupdatefiles={fileItems => {
            // Set currently active file objects to this.state
            this.setState({
              files: fileItems.map(fileItem => fileItem.file)
            });
          }}
        />
        <br></br>
        <table className="table table-dark">
            
     <tr>
        {headTDs}
    </tr>
    
    <tbody>
        {contextTD}
   </tbody>
          </table>

      </div>
   
    );
  }
}
const rootElement = document.getElementById('root')
ReactDOM.render(<App />, rootElement)
