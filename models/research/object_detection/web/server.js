const express = require('express')
const path = require('path')
const logger = require('morgan')
const cookieParser = require('cookie-parser')
const bodyParser = require('body-parser')
const fileUpload = require('express-fileupload')
const cors = require('cors')

const app = express()

// view engine setup
app.set('views', path.join(__dirname, 'views'))
app.set('view engine', 'jade')

// uncomment after placing your favicon in /public
//app.use(favicon(path.join(__dirname, 'public', 'favicon.ico')));
app.use(logger('dev'))
app.use(cors())
app.use(bodyParser.json())
app.use(
  bodyParser.urlencoded({
    extended: false,
  })
)
app.use(cookieParser())
app.use(fileUpload())
app.use('/public', express.static(__dirname + '/public'))

app.post('/upload', (req, res, next) => {
 console.log(req);
  let uploadFile = req.files.filepond;
  const fileName = req.files.filepond.name;
 
  uploadFile.mv(
    `/home/zyan11/Downloads/models/research/object_detection/test_images/${fileName}`,
      
    function (err) {
      if (err) {
        return res.status(500).send(err)
      }

      res.json({
        file: `${fileName}`,
      })
    }
  )
    console.log(`/home/zyan11/Downloads/models/research/object_detection/test_images/${fileName}`)
})

// catch 404 and forward to error handler
app.use(function (req, res, next) {
  const err = new Error('Not Found')
  err.status = 404
  next(err)
})

// error handler
app.use(function (err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message
  res.locals.error = req.app.get('env') === 'development' ? err : {}

  // render the error page
  res.status(err.status || 500)
  console.log(err)
})

 app.listen(3000, function () {
    console.log('lising on IPV4:127.0.0.1:3000');
});
