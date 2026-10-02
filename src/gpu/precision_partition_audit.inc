__global__ void audit_input_kernel(Complex* x, int nx, int ny, int nz,
    int px, int py, int x0, int y0, int z0, int global_n) {
  size_t row = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (row >= size_t(nx)*ny*nz) return;
  int i=row%nx, j=(row/nx)%ny, k=row/(size_t(nx)*ny);
  unsigned v=unsigned(x0+i)+unsigned(global_n)*(unsigned(y0+j)+unsigned(global_n)*unsigned(z0+k));
  v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15;
  double re=double(int(v&65535)-32768)/32768.0;
  double im=double(int((v>>16)&65535)-32768)/32768.0;
  x[padded_index(i+1,j+1,k+1,px,py)]=make_double2(re,im);
}

void audit_fill(BrickLevel& level, BrickVector& x) {
  x.zero();
  for (const auto& p:x.parts) {
    CK_CUDA(cudaSetDevice(p.device));
    audit_input_kernel<<<(p.interior_size()+255)/256,256>>>(p.input,
      p.nx,p.ny,p.nz,p.padded_nx(),p.padded_ny(),p.x0,p.y0,p.z0,level.params.nx);
    CK_CUDA(cudaGetLastError());
  }
  synchronize_parts(x.parts);
}

void audit_dump(const char* dir,const std::string& name,BrickLevel& level,
                const BrickVector& x) {
  MPI_File file;
  std::string path=std::string(dir)+"/"+name+".bin";
  CK_MPI(MPI_File_open(MPI_COMM_WORLD,path.c_str(),MPI_MODE_CREATE|MPI_MODE_WRONLY,
                       MPI_INFO_NULL,&file));
  CK_MPI(MPI_File_set_size(file,MPI_Offset(level.params.nx)*level.params.ny*level.params.nz*8));
  for(const auto& p:x.parts) {
    CK_CUDA(cudaSetDevice(p.device));
    std::vector<Complex> padded(size_t(p.padded_nx())*p.padded_ny()*(p.nz+2));
    CK_CUDA(cudaMemcpy(padded.data(),p.input,padded.size()*sizeof(Complex),cudaMemcpyDeviceToHost));
    std::vector<Complex> owned(p.interior_size());
    for(int k=0;k<p.nz;++k) for(int j=0;j<p.ny;++j) for(int i=0;i<p.nx;++i)
      owned[(size_t(k)*p.ny+j)*p.nx+i]=padded[(size_t(k+1)*p.padded_ny()+j+1)*p.padded_nx()+i+1];
    int sizes[4]={level.params.nz,level.params.ny,level.params.nx,2};
    int subs[4]={p.nz,p.ny,p.nx,2}, starts[4]={p.z0,p.y0,p.x0,0};
    MPI_Datatype view;
    CK_MPI(MPI_Type_create_subarray(4,sizes,subs,starts,MPI_ORDER_C,MPI_DOUBLE,&view));
    CK_MPI(MPI_Type_commit(&view));
    CK_MPI(MPI_File_set_view(file,0,MPI_DOUBLE,view,"native",MPI_INFO_NULL));
    CK_MPI(MPI_File_write_all(file,owned.data(),int(owned.size()*2),MPI_DOUBLE,MPI_STATUS_IGNORE));
    CK_MPI(MPI_Type_free(&view));
  }
  CK_MPI(MPI_File_close(&file));
}

void run_partition_audit(const char* dir,ThreeGrid& pc) {
  BrickLevel* levels[]={&pc.fine,&pc.coarse,&pc.coarse_shift,&pc.coarsest};
  for(int l=0;l<4;++l) {
    auto& level=*levels[l];
    BrickVector x(level),y(level);
    audit_fill(level,x);
    const std::string tag="L"+std::to_string(l);
    audit_dump(dir,tag+"_input",level,x);
    apply_operator(level,x,y);
    audit_dump(dir,tag+"_A",level,y);
    jacobi(level,x,y,l==3?pc.omega_shift_4h:pc.omega_shift_2h,2);
    audit_dump(dir,tag+"_J",level,y);
  }
  BrickVector x(pc.fine), y(pc.fine),c(pc.coarse),d(pc.coarsest);
  audit_fill(pc.fine,x);
  restrict_vector(pc.fine,x,c); audit_dump(dir,"R_h",pc.coarse,c);
  audit_fill(pc.coarse,c);
  prolong_vector(pc.coarse,c,y); audit_dump(dir,"P_h",pc.fine,y);
  restrict_vector(pc.coarse,c,d); audit_dump(dir,"R_2h",pc.coarsest,d);
  audit_fill(pc.coarsest,d);
  prolong_vector(pc.coarsest,d,c); audit_dump(dir,"P_2h",pc.coarse,c);
  audit_fill(pc.coarse,c);
  BrickVector s(pc.coarse);
  pc.shifted_two_grid(c,s); audit_dump(dir,"shifted_PC",pc.coarse,s);
  pc(x,y); audit_dump(dir,"full_PC",pc.fine,y);
}
